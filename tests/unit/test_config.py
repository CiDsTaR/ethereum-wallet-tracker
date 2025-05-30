"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from wallet_tracker.config import (
    AppConfig,
    CacheBackend,
    CoinGeckoConfig,
    Environment,
    EthereumConfig,
    GoogleSheetsConfig,
    Settings,
    SettingsError,
)


class TestEthereumConfig:
    """Test Ethereum configuration."""

    def test_valid_config(self) -> None:
        """Test valid Ethereum configuration."""
        config = EthereumConfig(
            alchemy_api_key="test_key",
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/test_key",
        )
        assert config.alchemy_api_key == "test_key"
        assert config.rate_limit == 100  # default

    def test_invalid_rpc_url(self) -> None:
        """Test invalid RPC URL validation."""
        with pytest.raises(ValidationError):
            EthereumConfig(alchemy_api_key="test_key", rpc_url="invalid_url")

    def test_rate_limit_validation(self) -> None:
        """Test rate limit validation."""
        with pytest.raises(ValidationError):
            EthereumConfig(
                alchemy_api_key="test_key",
                rpc_url="https://test.com",
                rate_limit=0,  # Invalid: too low
            )


class TestGoogleSheetsConfig:
    """Test Google Sheets configuration."""

    def test_valid_config_with_existing_file(self) -> None:
        """Test valid configuration with existing credentials file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "credentials"}')
            temp_path = Path(f.name)

        try:
            config = GoogleSheetsConfig(credentials_file=temp_path)
            assert config.credentials_file == temp_path
        finally:
            temp_path.unlink()

    def test_invalid_config_missing_file(self) -> None:
        """Test invalid configuration with missing credentials file."""
        with pytest.raises(ValidationError):
            GoogleSheetsConfig(credentials_file=Path("/nonexistent/file.json"))


class TestSettings:
    """Test settings management."""

    def test_env_loading_with_custom_file(self) -> None:
        """Test loading environment from custom file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_VAR=test_value\n")
            env_file = Path(f.name)

        try:
            Settings(env_file=env_file)
            # Should load the env file
            assert os.getenv("TEST_VAR") == "test_value"
        finally:
            env_file.unlink()
            # Clean up env var
            if "TEST_VAR" in os.environ:
                del os.environ["TEST_VAR"]

    @patch.dict(
        os.environ,
        {
            "ALCHEMY_API_KEY": "test_alchemy_key",
            "GOOGLE_SHEETS_CREDENTIALS_FILE": "test_credentials.json",
            "ENVIRONMENT": "development",
            "BATCH_SIZE": "25",
        },
    )
    def test_config_loading_from_env(self) -> None:
        """Test configuration loading from environment variables."""
        # Create a temporary credentials file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "credentials"}')
            temp_creds = Path(f.name)

        try:
            # Update env var to point to temp file
            os.environ["GOOGLE_SHEETS_CREDENTIALS_FILE"] = str(temp_creds)

            settings = Settings()
            config = settings.load_config()

            assert config.ethereum.alchemy_api_key == "test_alchemy_key"
            assert config.environment == Environment.DEVELOPMENT
            assert config.processing.batch_size == 25
            assert config.google_sheets.credentials_file == temp_creds

        finally:
            temp_creds.unlink()

    def test_missing_required_env_var(self) -> None:
        """Test error when required environment variable is missing."""
        # Create settings without loading .env file
        settings = Settings(env_file=Path("nonexistent.env"))

        with (
            patch.dict(os.environ, {}, clear=True),  # Clear all env vars
            pytest.raises(SettingsError, match="ALCHEMY_API_KEY"),
        ):
            settings.load_config()

    def test_config_validation_results(self) -> None:
        """Test configuration validation results."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "credentials"}')
            temp_creds = Path(f.name)

        try:
            with patch.dict(
                os.environ,
                {
                    "ALCHEMY_API_KEY": "test_key",
                    "GOOGLE_SHEETS_CREDENTIALS_FILE": str(temp_creds),
                    "ENVIRONMENT": "production",
                    "DEBUG": "true",  # This should generate a warning
                },
            ):
                settings = Settings()
                results = settings.validate_config()

                assert results["valid"] is True
                assert len(results["warnings"]) > 0  # Should warn about debug in production
                assert "Debug mode is enabled in production" in results["warnings"]

        finally:
            temp_creds.unlink()

    def test_config_caching(self) -> None:
        """Test that configuration is cached."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "credentials"}')
            temp_creds = Path(f.name)

        try:
            with patch.dict(
                os.environ,
                {
                    "ALCHEMY_API_KEY": "test_key",
                    "GOOGLE_SHEETS_CREDENTIALS_FILE": str(temp_creds),
                },
            ):
                settings = Settings()
                config1 = settings.get_config()
                config2 = settings.get_config()

                # Should be the same object (cached)
                assert config1 is config2

        finally:
            temp_creds.unlink()

    def test_config_reload(self) -> None:
        """Test configuration reloading."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "credentials"}')
            temp_creds = Path(f.name)

        try:
            with patch.dict(
                os.environ,
                {
                    "ALCHEMY_API_KEY": "test_key",
                    "GOOGLE_SHEETS_CREDENTIALS_FILE": str(temp_creds),
                    "BATCH_SIZE": "50",
                },
            ):
                settings = Settings()
                config1 = settings.get_config()
                assert config1.processing.batch_size == 50

                # Change environment and reload
                os.environ["BATCH_SIZE"] = "100"
                config2 = settings.reload_config()

                assert config2.processing.batch_size == 100
                assert config1 is not config2  # Should be different objects

        finally:
            temp_creds.unlink()


class TestAppConfig:
    """Test main application configuration."""

    def test_environment_helpers(self) -> None:
        """Test environment helper methods."""
        config = AppConfig(
            environment=Environment.PRODUCTION,
            ethereum=EthereumConfig(alchemy_api_key="test", rpc_url="https://test.com"),
            coingecko=CoinGeckoConfig(),
            google_sheets=GoogleSheetsConfig(
                credentials_file=Path(__file__)  # Use this test file as dummy
            ),
        )

        assert config.is_production() is True
        assert config.is_development() is False

        config.environment = Environment.DEVELOPMENT
        assert config.is_production() is False
        assert config.is_development() is True

    def test_cache_backend_helper(self) -> None:
        """Test cache backend helper method."""
        config = AppConfig(
            environment=Environment.DEVELOPMENT,
            ethereum=EthereumConfig(alchemy_api_key="test", rpc_url="https://test.com"),
            coingecko=CoinGeckoConfig(),
            google_sheets=GoogleSheetsConfig(credentials_file=Path(__file__)),
        )

        config.cache.backend = CacheBackend.REDIS
        assert config.get_cache_backend() == "redis"

        config.cache.backend = CacheBackend.FILE
        assert config.get_cache_backend() == "file"
