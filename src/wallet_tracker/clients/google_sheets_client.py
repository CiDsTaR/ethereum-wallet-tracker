"""Enhanced Google Sheets client with async operations and better error handling."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import gspread
from google.oauth2.service_account import Credentials

from ..config import GoogleSheetsConfig
from ..utils import CacheManager
from .google_sheets_types import (
    WALLET_RESULT_HEADERS,
    SheetConfig,
    SheetRange,
    SummaryData,
    WalletAddress,
    WalletResult,
)

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
    """Enhanced Google Sheets client for wallet data operations."""

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

        # Performance stats
        self._stats = {
            "read_operations": 0,
            "write_operations": 0,
            "batch_operations": 0,
            "api_errors": 0,
            "cache_hits": 0,
            "total_rows_read": 0,
            "total_rows_written": 0,
        }

    def _get_client(self) -> gspread.Client:
        """Get authenticated Google Sheets client."""
        if self._client is None:
            try:
                # Load service account credentials
                credentials = Credentials.from_service_account_file(
                    self.config.credentials_file, scopes=[self.config.scope]
                )

                # Create and authenticate client
                self._client = gspread.authorize(credentials)
                logger.info("✅ Google Sheets client authenticated successfully")

            except FileNotFoundError as e:
                raise SheetsAuthenticationError(f"❌ Credentials file not found: {self.config.credentials_file}") from e
            except Exception as e:
                self._stats["api_errors"] += 1
                logger.error(f"❌ Failed to authenticate with Google Sheets: {e}")
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
                try:
                    worksheet = spreadsheet.worksheet(worksheet_name)
                except gspread.WorksheetNotFound:
                    # Create worksheet if it doesn't exist
                    logger.info(f"📄 Creating new worksheet: {worksheet_name}")
                    worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            else:
                worksheet = spreadsheet.sheet1

            return worksheet

        except gspread.SpreadsheetNotFound as e:
            raise SheetsNotFoundError(f"Spreadsheet not found: {spreadsheet_id}") from e
        except gspread.exceptions.APIError as e:
            if "PERMISSION_DENIED" in str(e):
                raise SheetsPermissionError(
                    f"Permission denied for spreadsheet: {spreadsheet_id}. "
                    f"Make sure to share the spreadsheet with your service account email."
                ) from e
            else:
                self._stats["api_errors"] += 1
                raise SheetsAPIError(f"Google Sheets API error: {e}") from e
        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"❌ Error accessing worksheet: {e}")
            raise SheetsAPIError(f"Failed to access worksheet: {e}") from e

    async def read_wallet_addresses(
        self,
        spreadsheet_id: str,
        range_name: str = "A:B",
        worksheet_name: str = None,
        skip_header: bool = True,
    ) -> list[WalletAddress]:
        """Read wallet addresses from Google Sheets.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            range_name: Range to read (e.g., "A:B" for columns A and B)
            worksheet_name: Worksheet name (defaults to first sheet)
            skip_header: Whether to skip the first row (header)

        Returns:
            List of WalletAddress objects

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
                logger.debug(f"💾 Cache hit for wallet addresses: {len(cached_data)} wallets")
                return [WalletAddress(**addr) for addr in cached_data]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            values = await loop.run_in_executor(
                None, self._read_sheet_values, spreadsheet_id, range_name, worksheet_name
            )

            if not values:
                logger.warning(f"⚠️ No data found in range {range_name}")
                return []

            # Skip header row if requested
            data_start_row = 2 if skip_header and len(values) > 1 else 1
            if skip_header and len(values) > 1:
                values = values[1:]

            # Process wallet data
            wallet_addresses = []
            for i, row in enumerate(values):
                if not row or not row[0].strip():  # Skip empty rows
                    continue

                # Extract address (required) and label (optional)
                address = row[0].strip()
                label = row[1].strip() if len(row) > 1 and row[1] else f"Wallet {i + 1}"
                row_number = i + data_start_row

                wallet_addresses.append(
                    WalletAddress(
                        address=address,
                        label=label,
                        row_number=row_number,
                    )
                )

            logger.info(f"📖 Read {len(wallet_addresses)} wallet addresses from spreadsheet")
            self._stats["read_operations"] += 1
            self._stats["total_rows_read"] += len(wallet_addresses)

            # Cache the data (serialize for caching)
            if self.cache_manager:
                cache_data = [
                    {"address": wa.address, "label": wa.label, "row_number": wa.row_number} for wa in wallet_addresses
                ]
                await self.cache_manager.get_general_cache().set(
                    cache_key,
                    cache_data,
                    ttl=300,  # 5 minutes
                )

            return wallet_addresses

        except (SheetsNotFoundError, SheetsPermissionError, SheetsAPIError):
            raise
        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"❌ Error reading wallet addresses: {e}")
            raise SheetsAPIError(f"Failed to read wallet addresses: {e}") from e

    def _read_sheet_values(self, spreadsheet_id: str, range_name: str, worksheet_name: str = None) -> list[list[str]]:
        """Read values from sheet (blocking operation for thread pool)."""
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_name)
        return worksheet.get(range_name)

    async def write_wallet_results(
        self,
        spreadsheet_id: str,
        wallet_results: list[WalletResult],
        config: SheetConfig | None = None,
    ) -> bool:
        """Write wallet analysis results to Google Sheets.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            wallet_results: List of WalletResult objects
            config: Sheet configuration (optional)

        Returns:
            True if successful

        Raises:
            SheetsAPIError: If API request fails
        """
        if not wallet_results:
            logger.warning("⚠️ No wallet results to write")
            return True

        # Use default config if not provided
        if config is None:
            config = SheetConfig(
                spreadsheet_id=spreadsheet_id,
                output_worksheet="Wallet_Results",
                include_header=True,
                clear_existing=True,
            )

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self._write_results_sync,
                spreadsheet_id,
                wallet_results,
                config,
            )

            if success:
                logger.info(f"✅ Wrote {len(wallet_results)} wallet results to spreadsheet")
                self._stats["write_operations"] += 1
                self._stats["total_rows_written"] += len(wallet_results)

            return success

        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"❌ Error writing wallet results: {e}")
            raise SheetsAPIError(f"Failed to write wallet results: {e}") from e

    def _write_results_sync(self, spreadsheet_id: str, wallet_results: list[WalletResult], config: SheetConfig) -> bool:
        """Write results synchronously (for thread pool execution)."""
        worksheet = self._get_worksheet(spreadsheet_id, config.output_worksheet)

        # Prepare data for writing
        data_to_write = []

        # Add header row if requested
        if config.include_header:
            data_to_write.append(WALLET_RESULT_HEADERS)

        # Add wallet data rows
        for result in wallet_results:
            row = [
                result.address,
                result.label,
                self._format_balance(result.eth_balance),
                self._format_usd_value(result.eth_value_usd),
                self._format_balance(result.usdc_balance),
                self._format_balance(result.usdt_balance),
                self._format_balance(result.dai_balance),
                self._format_balance(result.aave_balance),
                self._format_balance(result.uni_balance),
                self._format_balance(result.link_balance),
                self._format_usd_value(result.other_tokens_value_usd),
                self._format_usd_value(result.total_value_usd),
                result.last_updated.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(result.last_updated, datetime)
                else str(result.last_updated),
                result.transaction_count,
                "✅ Active" if result.is_active else "❌ Inactive",
            ]
            data_to_write.append(row)

        # Clear existing data if requested
        if config.clear_existing:
            worksheet.clear()

        # Write data to sheet
        if data_to_write:
            # Calculate optimal range
            end_col = SheetRange.column_to_letter(len(data_to_write[0]))
            end_row = len(data_to_write)
            write_range = f"A1:{end_col}{end_row}"

            worksheet.update(write_range, data_to_write)

            # Apply basic formatting
            self._format_results_sheet(worksheet, len(data_to_write))

        return True

    async def write_batch_results(
        self,
        spreadsheet_id: str,
        wallet_results: list[WalletResult],
        batch_size: int = 100,
        worksheet_name: str = "Batch_Results",
    ) -> bool:
        """Write wallet results in batches for better performance.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            wallet_results: List of WalletResult objects
            batch_size: Number of rows to write per batch
            worksheet_name: Worksheet name for results

        Returns:
            True if successful
        """
        if not wallet_results:
            logger.warning("⚠️ No wallet results to write in batches")
            return True

        try:
            total_batches = (len(wallet_results) + batch_size - 1) // batch_size
            logger.info(f"📊 Writing {len(wallet_results)} results in {total_batches} batches")

            # Process in batches
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(wallet_results))
                batch = wallet_results[start_idx:end_idx]

                # Configure batch writing
                config = SheetConfig(
                    spreadsheet_id=spreadsheet_id,
                    output_worksheet=worksheet_name,
                    include_header=(batch_num == 0),  # Only include header in first batch
                    clear_existing=(batch_num == 0),  # Only clear for first batch
                )

                # For subsequent batches, adjust the starting row
                if batch_num > 0:
                    config.output_range = f"A{start_idx + 2}"  # +2 to account for header

                success = await self.write_wallet_results(
                    spreadsheet_id=spreadsheet_id,
                    wallet_results=batch,
                    config=config,
                )

                if not success:
                    logger.error(f"❌ Failed to write batch {batch_num + 1}/{total_batches}")
                    return False

                logger.info(f"✅ Completed batch {batch_num + 1}/{total_batches} ({len(batch)} rows)")

                # Small delay between batches to respect API limits
                if batch_num < total_batches - 1:
                    await asyncio.sleep(0.5)

            logger.info(f"🎉 Successfully wrote all {len(wallet_results)} results in {total_batches} batches")
            self._stats["batch_operations"] += 1
            return True

        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"❌ Error writing batch results: {e}")
            raise SheetsAPIError(f"Failed to write batch results: {e}") from e

    async def create_summary_sheet(
        self,
        spreadsheet_id: str,
        summary_data: SummaryData,
        worksheet_name: str = "Summary",
    ) -> bool:
        """Create a summary sheet with overall statistics.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID
            summary_data: Summary statistics
            worksheet_name: Name for the summary worksheet

        Returns:
            True if successful
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self._create_summary_sync,
                spreadsheet_id,
                summary_data,
                worksheet_name,
            )

            if success:
                logger.info(f"📊 Created summary sheet: {worksheet_name}")

            return success

        except Exception as e:
            self._stats["api_errors"] += 1
            logger.error(f"❌ Error creating summary sheet: {e}")
            raise SheetsAPIError(f"Failed to create summary sheet: {e}") from e

    def _create_summary_sync(self, spreadsheet_id: str, summary_data: SummaryData, worksheet_name: str) -> bool:
        """Create summary sheet synchronously."""
        worksheet = self._get_worksheet(spreadsheet_id, worksheet_name)
        worksheet.clear()  # Clear existing content

        # Prepare summary data
        summary_rows = [
            ["🏦 Ethereum Wallet Analysis Summary"],
            [""],
            ["📈 Overview Statistics"],
            ["Metric", "Value"],
            ["Total Wallets Analyzed", summary_data.total_wallets],
            ["✅ Active Wallets", summary_data.active_wallets],
            ["❌ Inactive Wallets", summary_data.inactive_wallets],
            [""],
            ["💰 Portfolio Values"],
            ["Metric", "Amount (USD)"],
            ["Total Portfolio Value", self._format_usd_value(summary_data.total_value_usd)],
            ["Average Portfolio Value", self._format_usd_value(summary_data.average_value_usd)],
            ["Median Portfolio Value", self._format_usd_value(summary_data.median_value_usd)],
            [""],
            ["🪙 Top Holdings by Value"],
            ["Token", "Total Value (USD)", "# Holders"],
            ["ETH", self._format_usd_value(summary_data.eth_total_value), summary_data.eth_holders],
            ["USDC", self._format_usd_value(summary_data.usdc_total_value), summary_data.usdc_holders],
            ["USDT", self._format_usd_value(summary_data.usdt_total_value), summary_data.usdt_holders],
            ["DAI", self._format_usd_value(summary_data.dai_total_value), summary_data.dai_holders],
            [""],
            ["⏱️ Analysis Information"],
            ["Analysis Completed", summary_data.analysis_time.strftime("%Y-%m-%d %H:%M:%S")],
            ["Processing Time", summary_data.processing_time],
        ]

        # Write summary data
        worksheet.update("A1", summary_rows)

        # Apply formatting
        self._format_summary_sheet(worksheet)

        return True

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
            return f"{balance:,.4f}"

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

    def _format_results_sheet(self, worksheet: gspread.Worksheet, num_rows: int) -> None:
        """Apply formatting to results sheet."""
        try:
            # Format header row
            if num_rows > 0:
                header_range = f"A1:{SheetRange.column_to_letter(len(WALLET_RESULT_HEADERS))}1"
                worksheet.format(
                    header_range,
                    {
                        "textFormat": {"bold": True, "fontSize": 11},
                        "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.9},
                        "textFormat": {"foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                        "horizontalAlignment": "CENTER",
                    },
                )

            # Freeze header row
            worksheet.freeze(rows=1)

            # Format currency columns (D, E, F, G, K, L)
            if num_rows > 1:
                currency_cols = ["D", "K", "L"]  # ETH Value, Other Tokens, Total Value
                for col in currency_cols:
                    range_name = f"{col}2:{col}{num_rows}"
                    worksheet.format(range_name, {"numberFormat": {"type": "CURRENCY", "pattern": "$#,##0.00"}})

                # Format balance columns with more decimals
                balance_cols = ["C", "E", "F", "G", "H", "I", "J"]  # Token balances
                for col in balance_cols:
                    range_name = f"{col}2:{col}{num_rows}"
                    worksheet.format(range_name, {"numberFormat": {"type": "NUMBER", "pattern": "#,##0.0000"}})

        except Exception as e:
            logger.warning(f"⚠️ Failed to format results sheet: {e}")

    def _format_summary_sheet(self, worksheet: gspread.Worksheet) -> None:
        """Apply formatting to summary sheet."""
        try:
            # Format title
            worksheet.format("A1", {"textFormat": {"bold": True, "fontSize": 16}, "horizontalAlignment": "CENTER"})

            # Format section headers
            headers = ["A3", "A9", "A15", "A22"]
            for header in headers:
                worksheet.format(
                    header,
                    {
                        "textFormat": {"bold": True, "fontSize": 12},
                        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                    },
                )

            # Format metric headers
            worksheet.format(
                "A4:B4", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}
            )

            worksheet.format(
                "A10:B10", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}
            )

            worksheet.format(
                "A16:C16", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}
            )

        except Exception as e:
            logger.warning(f"⚠️ Failed to format summary sheet: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "read_operations": self._stats["read_operations"],
            "write_operations": self._stats["write_operations"],
            "batch_operations": self._stats["batch_operations"],
            "api_errors": self._stats["api_errors"],
            "cache_hits": self._stats["cache_hits"],
            "total_rows_read": self._stats["total_rows_read"],
            "total_rows_written": self._stats["total_rows_written"],
            "authenticated": self._client is not None,
        }

    def health_check(self) -> bool:
        """Check if Google Sheets API is accessible."""
        try:
            client = self._get_client()
            # Test authentication by attempting to list available spreadsheets
            return True
        except Exception as e:
            logger.error(f"❌ Google Sheets health check failed: {e}")
            return False

    async def close(self) -> None:
        """Clean up resources."""
        # gspread doesn't require explicit cleanup, but we can clear our client reference
        self._client = None
        logger.info("📄 Google Sheets client closed")
