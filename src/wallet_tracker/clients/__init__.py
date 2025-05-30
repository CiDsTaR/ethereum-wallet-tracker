"""API clients package for the Ethereum Wallet Tracker.

This package provides clients for interacting with various APIs:
- Ethereum blockchain (via Alchemy)
- CoinGecko for token prices
- Google Sheets for data input/output
"""

from .ethereum_client import EthereumClient, EthereumClientError, InvalidAddressError, APIError
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

__all__ = [
    # Main client
    "EthereumClient",

    # Exceptions
    "EthereumClientError",
    "InvalidAddressError",
    "APIError",

    # Data types
    "TokenBalance",
    "EthBalance",
    "WalletPortfolio",
    "TokenMetadata",
    "TransactionInfo",
    "WalletActivity",

    # Utility functions
    "normalize_address",
    "is_valid_ethereum_address",
    "wei_to_eth",
    "format_token_amount",
    "calculate_token_value",
]