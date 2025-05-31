"""API clients package for the Ethereum Wallet Tracker.

This package provides clients for interacting with various APIs:
- Ethereum blockchain (via Alchemy)
- CoinGecko for token prices
- Google Sheets for data input/output
"""

from .ethereum_client import EthereumClient, EthereumClientError, InvalidAddressError, APIError as EthereumAPIError
from .ethereum_types import (
    TokenBalance,
    EthBalance,
    WalletPortfolio,
    TokenMetadata,
    TransactionInfo,
    WalletActivity,
    normalize_address,
    is_valid_ethereum_address,
    wei_to_eth,
    format_token_amount,
    calculate_token_value,
)

from .coingecko_client import (
    CoinGeckoClient,
    CoinGeckoPriceService,
    CoinGeckoClientError,
    APIError as CoinGeckoAPIError,
    RateLimitError,
)
from .coingecko_types import (
    TokenPrice,
    TokenSearchResult,
    ContractPriceResponse,
    get_coingecko_id,
    normalize_coingecko_price_data,
    is_stablecoin,
)

__all__ = [
    # Ethereum client
    "EthereumClient",
    "EthereumClientError",
    "InvalidAddressError",
    "EthereumAPIError",

    # CoinGecko client
    "CoinGeckoClient",
    "CoinGeckoPriceService",
    "CoinGeckoClientError",
    "CoinGeckoAPIError",
    "RateLimitError",

    # Ethereum data types
    "TokenBalance",
    "EthBalance",
    "WalletPortfolio",
    "TokenMetadata",
    "TransactionInfo",
    "WalletActivity",

    # CoinGecko data types
    "TokenPrice",
    "TokenSearchResult",
    "ContractPriceResponse",

    # Utility functions
    "normalize_address",
    "is_valid_ethereum_address",
    "wei_to_eth",
    "format_token_amount",
    "calculate_token_value",
    "get_coingecko_id",
    "normalize_coingecko_price_data",
    "is_stablecoin",
]