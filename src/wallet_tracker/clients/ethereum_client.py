"""Ethereum client for blockchain interactions using Alchemy APIs."""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import aiohttp
from asyncio_throttle import Throttler

from ..config import EthereumConfig
from ..utils import CacheManager
from .ethereum_types import (
    WELL_KNOWN_TOKENS,
    EthBalance,
    TokenBalance,
    WalletActivity,
    WalletPortfolio,
    calculate_token_value,
    format_token_amount,
    is_valid_ethereum_address,
    normalize_address,
    wei_to_eth,
)

logger = logging.getLogger(__name__)


class EthereumClientError(Exception):
    """Base exception for Ethereum client errors."""

    pass


class InvalidAddressError(EthereumClientError):
    """Invalid Ethereum address error."""

    pass


class APIError(EthereumClientError):
    """API request error."""

    pass


class EthereumClient:
    """Ethereum client using Alchemy Portfolio API and Web3."""

    def __init__(
        self,
        config: EthereumConfig,
        cache_manager: CacheManager | None = None,
        session: aiohttp.ClientSession | None = None,
    ):
        """Initialize Ethereum client.

        Args:
            config: Ethereum configuration
            cache_manager: Cache manager for caching responses
            session: HTTP session (optional, will create if not provided)
        """
        self.config = config
        self.cache_manager = cache_manager
        self._session = session
        self._own_session = session is None

        # Rate limiting
        self.throttler = Throttler(rate_limit=config.rate_limit, period=60)

        # API endpoints
        self.portfolio_url = f"{config.rpc_url.replace('/v2/', '/v2/getTokenBalances')}"
        self.metadata_url = f"{config.rpc_url.replace('/v2/', '/v2/getTokenMetadata')}"
        self.transaction_url = f"{config.rpc_url.replace('/v2/', '/v2/getAssetTransfers')}"
        self.eth_balance_url = f"{config.rpc_url.replace('/v2/', '/v2/eth_getBalance')}"

        # Stats
        self._stats = {
            "portfolio_requests": 0,
            "metadata_requests": 0,
            "transaction_requests": 0,
            "cache_hits": 0,
            "api_errors": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session is available."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "EthereumWalletTracker/1.0",
                },
            )
        return self._session

    async def _make_request(
        self,
        method: str,
        url: str,
        data: dict[str, Any] | None = None,
        cache_key: str | None = None,
        cache_ttl: int | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with rate limiting and caching.

        Args:
            method: HTTP method
            url: Request URL
            data: Request data
            cache_key: Cache key for response
            cache_ttl: Cache TTL in seconds

        Returns:
            Response data

        Raises:
            APIError: If request fails
        """
        # Check cache first
        if cache_key and self.cache_manager:
            cached_response = await self.cache_manager.get_general_cache().get(cache_key)
            if cached_response:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_response

        # Rate limiting
        async with self.throttler:
            session = await self._ensure_session()

            try:
                if method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        response_data = await self._handle_response(response)
                else:
                    async with session.get(url, params=data) as response:
                        response_data = await self._handle_response(response)

                # Cache successful response
                if cache_key and self.cache_manager and cache_ttl:
                    await self.cache_manager.get_general_cache().set(cache_key, response_data, ttl=cache_ttl)

                return response_data

            except Exception as e:
                self._stats["api_errors"] += 1
                logger.error(f"API request failed: {e}")
                raise APIError(f"Request to {url} failed: {e}") from e

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Handle HTTP response and extract data."""
        if response.status == 200:
            data = await response.json()

            # Check for Alchemy API errors
            if "error" in data:
                error_msg = data["error"].get("message", "Unknown API error")
                raise APIError(f"Alchemy API error: {error_msg}")

            return data

        elif response.status == 429:
            # Rate limited
            raise APIError("Rate limited by Alchemy API")

        else:
            error_text = await response.text()
            raise APIError(f"HTTP {response.status}: {error_text}")

    async def get_wallet_portfolio(
        self,
        wallet_address: str,
        include_metadata: bool = True,
        include_prices: bool = True,
    ) -> WalletPortfolio:
        """Get complete wallet portfolio using Alchemy Portfolio API.

        Args:
            wallet_address: Ethereum wallet address
            include_metadata: Whether to fetch token metadata
            include_prices: Whether to fetch token prices

        Returns:
            Complete wallet portfolio

        Raises:
            InvalidAddressError: If wallet address is invalid
            APIError: If API request fails
        """
        # Validate address
        if not is_valid_ethereum_address(wallet_address):
            raise InvalidAddressError(f"Invalid Ethereum address: {wallet_address}")

        normalized_address = normalize_address(wallet_address)

        # Check cache first
        cache_key = f"portfolio:{normalized_address}"
        if self.cache_manager:
            cached_portfolio = await self.cache_manager.get_balance(normalized_address)
            if cached_portfolio:
                self._stats["cache_hits"] += 1
                logger.debug(f"Portfolio cache hit for {normalized_address}")
                return self._deserialize_portfolio(cached_portfolio)

        logger.info(f"Fetching portfolio for wallet: {normalized_address}")

        # Get ETH balance
        eth_balance = await self._get_eth_balance(normalized_address)

        # Get token balances using Portfolio API
        token_balances = await self._get_token_balances(normalized_address, include_metadata, include_prices)

        # Get wallet activity
        activity = await self._get_wallet_activity(normalized_address)

        # Calculate total value
        total_value_usd = eth_balance.value_usd or Decimal("0")
        for token in token_balances:
            if token.value_usd:
                total_value_usd += token.value_usd

        # Create portfolio
        portfolio = WalletPortfolio(
            address=normalized_address,
            eth_balance=eth_balance,
            token_balances=token_balances,
            total_value_usd=total_value_usd,
            last_updated=datetime.now(UTC),
            transaction_count=activity.transaction_count,
            last_transaction_hash=activity.last_transaction.hash if activity.last_transaction else None,
            last_transaction_timestamp=activity.last_transaction.timestamp if activity.last_transaction else None,
        )

        # Cache the portfolio
        if self.cache_manager:
            serialized = self._serialize_portfolio(portfolio)
            await self.cache_manager.set_balance(normalized_address, serialized)

        self._stats["portfolio_requests"] += 1
        return portfolio

    async def _get_eth_balance(self, wallet_address: str) -> EthBalance:
        """Get ETH balance for wallet."""
        data = {"id": 1, "jsonrpc": "2.0", "method": "eth_getBalance", "params": [wallet_address, "latest"]}

        cache_key = f"eth_balance:{wallet_address}"
        response = await self._make_request(
            "POST",
            self.eth_balance_url,
            data=data,
            cache_key=cache_key,
            cache_ttl=300,  # 5 minutes
        )

        balance_wei = response["result"]
        balance_eth = wei_to_eth(balance_wei)

        # Get ETH price (you might want to integrate with CoinGecko here)
        eth_price_usd = await self._get_eth_price()
        value_usd = balance_eth * eth_price_usd if eth_price_usd else None

        return EthBalance(
            balance_wei=balance_wei,
            balance_eth=balance_eth,
            price_usd=eth_price_usd,
            value_usd=value_usd,
        )

    async def _get_token_balances(
        self,
        wallet_address: str,
        include_metadata: bool,
        include_prices: bool,
    ) -> list[TokenBalance]:
        """Get token balances using Alchemy getTokenBalances API."""
        data = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getTokenBalances", "params": [wallet_address]}

        cache_key = f"token_balances:{wallet_address}"
        response = await self._make_request(
            "POST",
            self.portfolio_url,
            data=data,
            cache_key=cache_key,
            cache_ttl=600,  # 10 minutes
        )

        token_balances = []
        token_data = response["result"]["tokenBalances"]

        # Get metadata for all tokens if requested
        metadata_map = {}
        if include_metadata and token_data:
            contract_addresses = [token["contractAddress"] for token in token_data if token["tokenBalance"] != "0x0"]
            metadata_map = await self._get_tokens_metadata(contract_addresses)

        # Get prices for all tokens if requested
        price_map = {}
        if include_prices and token_data:
            contract_addresses = [token["contractAddress"] for token in token_data if token["tokenBalance"] != "0x0"]
            price_map = await self._get_tokens_prices(contract_addresses)

        for token_data in token_data:
            # Skip zero balances
            if token_data["tokenBalance"] == "0x0":
                continue

            contract_address = normalize_address(token_data["contractAddress"])
            raw_balance = str(int(token_data["tokenBalance"], 16))

            # Get metadata
            metadata = metadata_map.get(contract_address) or WELL_KNOWN_TOKENS.get(contract_address, {})
            if not metadata:
                # Skip tokens without metadata
                continue

            # Format balance
            decimals = metadata.get("decimals", 18)
            formatted_balance = format_token_amount(raw_balance, decimals)

            # Calculate value
            price_usd = price_map.get(contract_address)
            value_usd = calculate_token_value(formatted_balance, price_usd)

            token_balance = TokenBalance(
                contract_address=contract_address,
                symbol=metadata.get("symbol", "UNKNOWN"),
                name=metadata.get("name", "Unknown Token"),
                decimals=decimals,
                balance_raw=raw_balance,
                balance_formatted=formatted_balance,
                price_usd=price_usd,
                value_usd=value_usd,
                logo_url=metadata.get("logo"),
                is_verified=metadata.get("is_verified", False),
            )

            token_balances.append(token_balance)

        return token_balances

    async def _get_tokens_metadata(self, contract_addresses: list[str]) -> dict[str, dict[str, Any]]:
        """Get metadata for multiple tokens."""
        if not contract_addresses:
            return {}

        # Check cache first
        metadata_map = {}
        uncached_addresses = []

        if self.cache_manager:
            for address in contract_addresses:
                cached_metadata = await self.cache_manager.get_token_metadata(address)
                if cached_metadata:
                    metadata_map[address] = cached_metadata
                else:
                    uncached_addresses.append(address)
        else:
            uncached_addresses = contract_addresses

        # Fetch uncached metadata
        if uncached_addresses:
            data = {"id": 1, "jsonrpc": "2.0", "method": "alchemy_getTokenMetadata", "params": uncached_addresses}

            try:
                response = await self._make_request("POST", self.metadata_url, data=data)
                self._stats["metadata_requests"] += 1

                # Process response
                for i, token_metadata in enumerate(response.get("result", [])):
                    if i < len(uncached_addresses):
                        address = normalize_address(uncached_addresses[i])
                        metadata = {
                            "symbol": token_metadata.get("symbol"),
                            "name": token_metadata.get("name"),
                            "decimals": token_metadata.get("decimals", 18),
                            "logo": token_metadata.get("logo"),
                        }
                        metadata_map[address] = metadata

                        # Cache metadata
                        if self.cache_manager:
                            await self.cache_manager.set_token_metadata(address, metadata)

            except Exception as e:
                logger.warning(f"Failed to fetch token metadata: {e}")

        return metadata_map

    async def _get_tokens_prices(self, contract_addresses: list[str]) -> dict[str, Decimal]:
        """Get prices for multiple tokens."""
        # This is a placeholder - in real implementation, you'd integrate with CoinGecko
        # For now, return empty dict
        return {}

    async def _get_eth_price(self) -> Decimal | None:
        """Get current ETH price in USD."""
        # This is a placeholder - integrate with CoinGecko client
        return Decimal("2000.0")  # Dummy price

    async def _get_wallet_activity(self, wallet_address: str) -> WalletActivity:
        """Get wallet activity summary."""
        # Check cache first
        if self.cache_manager:
            cached_activity = await self.cache_manager.get_wallet_activity(wallet_address)
            if cached_activity:
                return self._deserialize_activity(cached_activity)

        # This is a simplified implementation
        # In production, you'd fetch recent transactions and analyze activity
        activity = WalletActivity(
            address=wallet_address,
            first_transaction=None,
            last_transaction=None,
            transaction_count=0,
            total_gas_used=0,
            is_active=True,  # Will be determined by actual transaction analysis
        )

        # Cache activity
        if self.cache_manager:
            serialized = self._serialize_activity(activity)
            await self.cache_manager.set_wallet_activity(wallet_address, serialized)

        return activity

    def _serialize_portfolio(self, portfolio: WalletPortfolio) -> dict[str, Any]:
        """Serialize portfolio for caching."""
        return {
            "address": portfolio.address,
            "eth_balance": {
                "balance_wei": portfolio.eth_balance.balance_wei,
                "balance_eth": str(portfolio.eth_balance.balance_eth),
                "price_usd": str(portfolio.eth_balance.price_usd) if portfolio.eth_balance.price_usd else None,
                "value_usd": str(portfolio.eth_balance.value_usd) if portfolio.eth_balance.value_usd else None,
            },
            "token_balances": [
                {
                    "contract_address": token.contract_address,
                    "symbol": token.symbol,
                    "name": token.name,
                    "decimals": token.decimals,
                    "balance_raw": token.balance_raw,
                    "balance_formatted": str(token.balance_formatted),
                    "price_usd": str(token.price_usd) if token.price_usd else None,
                    "value_usd": str(token.value_usd) if token.value_usd else None,
                    "logo_url": token.logo_url,
                    "is_verified": token.is_verified,
                }
                for token in portfolio.token_balances
            ],
            "total_value_usd": str(portfolio.total_value_usd),
            "last_updated": portfolio.last_updated.isoformat(),
            "transaction_count": portfolio.transaction_count,
        }

    def _deserialize_portfolio(self, data: dict[str, Any]) -> WalletPortfolio:
        """Deserialize portfolio from cache."""
        eth_data = data["eth_balance"]
        eth_balance = EthBalance(
            balance_wei=eth_data["balance_wei"],
            balance_eth=Decimal(eth_data["balance_eth"]),
            price_usd=Decimal(eth_data["price_usd"]) if eth_data["price_usd"] else None,
            value_usd=Decimal(eth_data["value_usd"]) if eth_data["value_usd"] else None,
        )

        token_balances = []
        for token_data in data["token_balances"]:
            token_balance = TokenBalance(
                contract_address=token_data["contract_address"],
                symbol=token_data["symbol"],
                name=token_data["name"],
                decimals=token_data["decimals"],
                balance_raw=token_data["balance_raw"],
                balance_formatted=Decimal(token_data["balance_formatted"]),
                price_usd=Decimal(token_data["price_usd"]) if token_data["price_usd"] else None,
                value_usd=Decimal(token_data["value_usd"]) if token_data["value_usd"] else None,
                logo_url=token_data["logo_url"],
                is_verified=token_data["is_verified"],
            )
            token_balances.append(token_balance)

        return WalletPortfolio(
            address=data["address"],
            eth_balance=eth_balance,
            token_balances=token_balances,
            total_value_usd=Decimal(data["total_value_usd"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            transaction_count=data["transaction_count"],
        )

    def _serialize_activity(self, activity: WalletActivity) -> dict[str, Any]:
        """Serialize activity for caching."""
        return {
            "address": activity.address,
            "transaction_count": activity.transaction_count,
            "total_gas_used": activity.total_gas_used,
            "is_active": activity.is_active,
            "days_since_last_transaction": activity.days_since_last_transaction,
        }

    def _deserialize_activity(self, data: dict[str, Any]) -> WalletActivity:
        """Deserialize activity from cache."""
        return WalletActivity(
            address=data["address"],
            first_transaction=None,  # Not cached for now
            last_transaction=None,  # Not cached for now
            transaction_count=data["transaction_count"],
            total_gas_used=data["total_gas_used"],
            is_active=data["is_active"],
            days_since_last_transaction=data.get("days_since_last_transaction"),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "portfolio_requests": self._stats["portfolio_requests"],
            "metadata_requests": self._stats["metadata_requests"],
            "transaction_requests": self._stats["transaction_requests"],
            "cache_hits": self._stats["cache_hits"],
            "api_errors": self._stats["api_errors"],
            "rate_limit": self.config.rate_limit,
        }

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and self._own_session:
            await self._session.close()
            self._session = None
            logger.info("Ethereum client session closed")
