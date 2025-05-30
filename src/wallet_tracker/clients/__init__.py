"""API clients package for the Ethereum Wallet Tracker.

This package provides clients for interacting with various APIs:
- Ethereum blockchain (via Alchemy)
- CoinGecko for token prices
- Google Sheets for data input/output
"""

from .ethereum_client import APIError, EthereumClient, EthereumClientError, InvalidAddressError
from .ethereum_types import (
    EthBalance,
    TokenBalance,
    TokenMetadata,
    TransactionInfo,
    WalletActivity,
    WalletPortfolio,
    calculate_token_value,
    format_token_amount,
    is_valid_ethereum_address,
    normalize_address,
    wei_to_eth,
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
