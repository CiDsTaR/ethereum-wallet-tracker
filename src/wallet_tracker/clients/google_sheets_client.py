"""Google Sheets client for reading wallet addresses and writing results."""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import gspread
from google.oauth2.service_account import Credentials

from ..config import GoogleSheetsConfig
from ..utils import CacheManager

logger = logging.getLogger(__name__)


class GoogleSheetsClientError(Exception):
    """Base exception for Google Sheets client errors."""

    pass


class SheetsAuthenticationError(GoogleSheetsClientError):
    """Google Sheets authentication error."""

    pass


class SheetsNotFoundError(GoogleSheetsClientError):
    """Google Sheets spreadsheet or worksheet not found error."""

    pass


class SheetsPermissionError(GoogleSheetsClientError):
    """Google Sheets permission error."""

    pass


class SheetsAPIError(GoogleSheetsClientError):
    """Google Sheets API error."""

    pass


class GoogleSheetsClient:
    """Google Sheets client for wallet data input/output operations."""

    def __init__(
        self,
        config: GoogleSheetsConfig,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize Google Sheets client.

        Args:
            config: Google Sheets configuration
            cache_manager: Cache manager for caching responses
        """
        self.config = config
        self.cache_manager = cache_manager
        self._client: gspread.Client | None = None

        # Stats
        self._stats = {
            "read_operations": 0,
            "write_operations": 0,
            "batch_operations": 0,
            "api_errors": 0,
            "cache_hits": 0,
        }

    def _get_client(self) -> gspread.Client:
        """Get authenticated Google Sheets client."""
        if self._client is None:
            try:
                # Load credentials
                credentials = Credentials.from_service_account_file(
                    self.config.credentials_file,
                    scopes=[self.config.scope]
                )

                # Create client
                self._client = gspread.authorize(credentials)
                logger.info("Google Sheets client authenticated successfully")

            except FileNotFoundError as e:
                raise SheetsAuthenticationError(
                    f"Credentials file not found: {self.config.credentials_file}"
                ) from e
            except Exception as e:
                self._stats["api_errors"] += 1
                logger.error(f"Failed to authenticate with Google Sheets: {e}")
                raise SheetsAuthenticationError(f"Google Sheets authentication failed: {e}") from e

        return self._client

    def _get_worksheet(self, spreadsheet_id: str, worksheet_name: str = None) -> gspread.Worksheet:
        """Get worksheet from spreadsheet.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            worksheet_name: Worksheet name (defaults to first sheet)

        Returns:
            Worksheet object

        Raises:
            SheetsNotFoundError: If spreadsheet or worksheet not found
            SheetsPermissionError: If permission denied
        """
        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(spreadsheet_id)

            if worksheet_name:
                worksheet = spreadsheet.worksheet(worksheet_name)
            else:
                worksheet = spreadsheet.sheet1

            return worksheet

        except gspread.SpreadsheetNotFound as e:
            raise SheetsNotFoundError(f"Spreadsheet not found: {spreadsheet_id}") from e
        except gspread.WorksheetNotFound as e:
            raise SheetsNotFoundError(f"Worksheet not found: {worksheet_name}") from e
        except gspread.exceptions.APIError as e:
            if "PERMISSION_DENIED" in str(e):
                raise SheetsPermissionError(
                    f"Permission denied for spreadsheet: {spreadsheet_id}"
                ) from e
            else:
                self._stats["api_errors"] += 1
                raise SheetsAPIError(f"Google Sheets API error: {e}") from e
        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"Error accessing worksheet: {e}")
            raise SheetsAPIError(f"Failed to access worksheet: {e}") from e

    async def read_wallet_addresses(
        self,
        spreadsheet_id: str,
        range_name: str = "A:B",
        worksheet_name: str = None,
        skip_header: bool = True,
    ) -> list[dict[str, str]]:
        """Read wallet addresses from Google Sheets.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            range_name: Range to read (e.g., "A:B" for columns A and B)
            worksheet_name: Worksheet name (defaults to first sheet)
            skip_header: Whether to skip the first row (header)

        Returns:
            List of dictionaries with 'address' and 'label' keys

        Raises:
            SheetsNotFoundError: If spreadsheet or worksheet not found
            SheetsAPIError: If API request fails
        """
        cache_key = f"wallet_addresses:{spreadsheet_id}:{range_name}:{worksheet_name}"

        # Check cache first
        if self.cache_manager:
            cached_data = await self.cache_manager.get_general_cache().get(cache_key)
            if cached_data:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for wallet addresses: {cache_key}")
                return cached_data

        try:
            worksheet = self._get_worksheet(spreadsheet_id, worksheet_name)

            # Get values from range
            values = worksheet.get(range_name)

            if not values:
                logger.warning(f"No data found in range {range_name}")
                return []

            # Skip header row if requested
            if skip_header and len(values) > 1:
                values = values[1:]

            # Process wallet data
            wallet_data = []
            for i, row in enumerate(values, start=2 if skip_header else 1):
                if not row:  # Skip empty rows
                    continue

                # Extract address (required) and label (optional)
                address = row[0].strip() if row and row[0] else ""
                label = row[1].strip() if len(row) > 1 and row[1] else f"Wallet {i}"

                if address:  # Only include rows with addresses
                    wallet_data.append({
                        "address": address,
                        "label": label,
                        "row_number": i,
                    })

            logger.info(f"Read {len(wallet_data)} wallet addresses from spreadsheet")

            # Cache the data
            if self.cache_manager:
                await self.cache_manager.get_general_cache().set(
                    cache_key, wallet_data, ttl=300  # 5 minutes
                )

            self._stats["read_operations"] += 1
            return wallet_data

        except (SheetsNotFoundError, SheetsPermissionError, SheetsAPIError):
            raise
        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"Error reading wallet addresses: {e}")
            raise SheetsAPIError(f"Failed to read wallet addresses: {e}") from e

    def write_wallet_results(
        self,
        spreadsheet_id: str,
        wallet_results: list[dict[str, Any]],
        range_start: str = "A1",
        worksheet_name: str = None,
        include_header: bool = True,
        clear_existing: bool = True,
    ) -> bool:
        """Write wallet analysis results to Google Sheets.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            wallet_results: List of wallet result dictionaries
            range_start: Starting cell for writing (e.g., "A1")
            worksheet_name: Worksheet name (defaults to first sheet)
            include_header: Whether to include header row
            clear_existing: Whether to clear existing data

        Returns:
            True if successful

        Raises:
            SheetsNotFoundError: If spreadsheet or worksheet not found
            SheetsAPIError: If API request fails
        """
        if not wallet_results:
            logger.warning("No wallet results to write")
            return True

        try:
            worksheet = self._get_worksheet(spreadsheet_id, worksheet_name)

            # Prepare data for writing
            data_to_write = []

            # Add header row if requested
            if include_header:
                header = [
                    "Address",
                    "Label",
                    "ETH Balance",
                    "ETH Value (USD)",
                    "USDC Balance",
                    "USDT Balance",
                    "DAI Balance",
                    "AAVE Balance",
                    "UNI Balance",
                    "LINK Balance",
                    "Other Tokens Value (USD)",
                    "Total Value (USD)",
                    "Last Updated",
                    "Transaction Count",
                    "Is Active",
                ]
                data_to_write.append(header)

            # Add wallet data rows
            for result in wallet_results:
                row = [
                    result.get("address", ""),
                    result.get("label", ""),
                    self._format_balance(result.get("eth_balance", 0)),
                    self._format_usd_value(result.get("eth_value_usd", 0)),
                    self._format_balance(result.get("usdc_balance", 0)),
                    self._format_balance(result.get("usdt_balance", 0)),
                    self._format_balance(result.get("dai_balance", 0)),
                    self._format_balance(result.get("aave_balance", 0)),
                    self._format_balance(result.get("uni_balance", 0)),
                    self._format_balance(result.get("link_balance", 0)),
                    self._format_usd_value(result.get("other_tokens_value_usd", 0)),
                    self._format_usd_value(result.get("total_value_usd", 0)),
                    result.get("last_updated", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")),
                    result.get("transaction_count", 0),
                    "Yes" if result.get("is_active", False) else "No",
                ]
                data_to_write.append(row)

            # Clear existing data if requested
            if clear_existing:
                # Calculate range to clear based on data size
                end_col = chr(ord('A') + len(data_to_write[0]) - 1)
                end_row = len(data_to_write) + 10  # Clear a bit more to ensure old data is removed
                clear_range = f"{range_start}:{end_col}{end_row}"
                worksheet.batch_clear([clear_range])

            # Write data to sheet
            if data_to_write:
                # Calculate the range for writing
                end_col = chr(ord(range_start[0]) + len(data_to_write[0]) - 1)
                end_row = int(range_start[1:]) + len(data_to_write) - 1
                write_range = f"{range_start}:{end_col}{end_row}"

                worksheet.update(write_range, data_to_write)

            logger.info(f"Wrote {len(wallet_results)} wallet results to spreadsheet")
            self._stats["write_operations"] += 1
            return True

        except (SheetsNotFoundError, SheetsPermissionError, SheetsAPIError):
            raise
        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"Error writing wallet results: {e}")
            raise SheetsAPIError(f"Failed to write wallet results: {e}") from e

    def write_batch_results(
        self,
        spreadsheet_id: str,
        batch_results: list[list[dict[str, Any]]],
        worksheet_name: str = None,
        batch_size: int = 100,
    ) -> bool:
        """Write wallet results in batches for better performance.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            batch_results: List of batches of wallet results
            worksheet_name: Worksheet name (defaults to first sheet)
            batch_size: Number of rows to write per batch

        Returns:
            True if successful

        Raises:
            SheetsAPIError: If API request fails
        """
        try:
            # Flatten batch results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

            if not all_results:
                logger.warning("No batch results to write")
                return True

            # Write in chunks for better performance
            total_written = 0
            for i in range(0, len(all_results), batch_size):
                chunk = all_results[i:i + batch_size]

                # For first chunk, include header and clear existing data
                include_header = (i == 0)
                clear_existing = (i == 0)

                # Calculate starting row for this chunk
                start_row = 1 if i == 0 else (i + 2)  # +2 to account for header
                range_start = f"A{start_row}"

                success = self.write_wallet_results(
                    spreadsheet_id=spreadsheet_id,
                    wallet_results=chunk,
                    range_start=range_start,
                    worksheet_name=worksheet_name,
                    include_header=include_header,
                    clear_existing=clear_existing,
                )

                if success:
                    total_written += len(chunk)
                    logger.info(f"Wrote batch {i//batch_size + 1}: {len(chunk)} rows")
                else:
                    logger.error(f"Failed to write batch {i//batch_size + 1}")
                    return False

            logger.info(f"Successfully wrote {total_written} total wallet results in batches")
            self._stats["batch_operations"] += 1
            return True

        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"Error writing batch results: {e}")
            raise SheetsAPIError(f"Failed to write batch results: {e}") from e

    def create_summary_sheet(
        self,
        spreadsheet_id: str,
        summary_data: dict[str, Any],
        worksheet_name: str = "Summary",
    ) -> bool:
        """Create a summary sheet with overall statistics.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            summary_data: Summary statistics dictionary
            worksheet_name: Name for the summary worksheet

        Returns:
            True if successful

        Raises:
            SheetsAPIError: If API request fails
        """
        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(spreadsheet_id)

            # Try to get existing summary worksheet or create new one
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                worksheet.clear()  # Clear existing content
            except gspread.WorksheetNotFound:
                worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=50, cols=10)

            # Prepare summary data
            summary_rows = [
                ["Ethereum Wallet Analysis Summary"],
                [""],
                ["Metric", "Value"],
                [""],
                ["Total Wallets Analyzed", summary_data.get("total_wallets", 0)],
                ["Active Wallets", summary_data.get("active_wallets", 0)],
                ["Inactive Wallets", summary_data.get("inactive_wallets", 0)],
                ["Total Portfolio Value (USD)", self._format_usd_value(summary_data.get("total_value_usd", 0))],
                ["Average Portfolio Value (USD)", self._format_usd_value(summary_data.get("average_value_usd", 0))],
                ["Median Portfolio Value (USD)", self._format_usd_value(summary_data.get("median_value_usd", 0))],
                [""],
                ["Top Holdings by Value"],
                ["Token", "Total Value (USD)", "Wallets Holding"],
                ["ETH", self._format_usd_value(summary_data.get("eth_total_value", 0)), summary_data.get("eth_holders", 0)],
                ["USDC", self._format_usd_value(summary_data.get("usdc_total_value", 0)), summary_data.get("usdc_holders", 0)],
                ["USDT", self._format_usd_value(summary_data.get("usdt_total_value", 0)), summary_data.get("usdt_holders", 0)],
                ["DAI", self._format_usd_value(summary_data.get("dai_total_value", 0)), summary_data.get("dai_holders", 0)],
                [""],
                ["Analysis Completed", summary_data.get("analysis_time", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"))],
                ["Processing Time", summary_data.get("processing_time", "Unknown")],
            ]

            # Write summary data
            worksheet.update("A1", summary_rows)

            # Format the summary sheet
            self._format_summary_sheet(worksheet)

            logger.info(f"Created summary sheet: {worksheet_name}")
            return True

        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"Error creating summary sheet: {e}")
            raise SheetsAPIError(f"Failed to create summary sheet: {e}") from e

    def _format_balance(self, balance: Decimal | float | str) -> str:
        """Format token balance for display."""
        if isinstance(balance, str):
            try:
                balance = Decimal(balance)
            except:
                return str(balance)

        if isinstance(balance, (int, float)):
            balance = Decimal(str(balance))

        if balance == 0:
            return "0"
        elif balance < Decimal("0.001"):
            return f"{balance:.6f}"
        elif balance < Decimal("1"):
            return f"{balance:.4f}"
        else:
            return f"{balance:.2f}"

    def _format_usd_value(self, value: Decimal | float | str) -> str:
        """Format USD value for display."""
        if isinstance(value, str):
            try:
                value = Decimal(value)
            except:
                return str(value)

        if isinstance(value, (int, float)):
            value = Decimal(str(value))

        if value == 0:
            return "$0.00"
        elif value < Decimal("0.01"):
            return f"${value:.4f}"
        else:
            return f"${value:,.2f}"

    def _format_summary_sheet(self, worksheet: gspread.Worksheet) -> None:
        """Apply formatting to summary sheet."""
        try:
            # Format title
            worksheet.format("A1", {
                "textFormat": {"bold": True, "fontSize": 14},
                "horizontalAlignment": "CENTER"
            })

            # Format headers
            worksheet.format("A3:B3", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })

            worksheet.format("A12:C12", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}
            })

        except Exception as e:
            logger.warning(f"Failed to format summary sheet: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "read_operations": self._stats["read_operations"],
            "write_operations": self._stats["write_operations"],
            "batch_operations": self._stats["batch_operations"],
            "api_errors": self._stats["api_errors"],
            "cache_hits": self._stats["cache_hits"],
            "authenticated": self._client is not None,
        }

    def health_check(self) -> bool:
        """Check if Google Sheets API is accessible."""
        try:
            client = self._get_client()
            # Try to list spreadsheets to test authentication
            # Note: This requires the Drive API scope to work
            return True
        except Exception as e:
            logger.error(f"Google Sheets health check failed: {e}")
            return False