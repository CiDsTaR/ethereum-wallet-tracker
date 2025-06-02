"""Basic usage examples for the Ethereum Wallet Tracker.

This file demonstrates:
- Simple wallet analysis
- Getting started guide
- Common patterns and best practices
- Basic error handling
"""

import asyncio
import logging
from decimal import Decimal
from typing import List, Dict, Any

from wallet_tracker.app import Application, create_application
from wallet_tracker.config import get_config, AppConfig
from wallet_tracker.clients import (
    EthereumClientError,
    CoinGeckoClientError,
    GoogleSheetsClientError,
    InvalidAddressError
)

# Setup logging for examples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_simple_wallet_analysis():
    """Example 1: Analyze a single wallet address."""

    print("🏦 Example 1: Simple Wallet Analysis")
    print("=" * 50)

    # The wallet address to analyze
    wallet_address = "0x742d35Cc6634C0532925a3b8D40e4f337F42090B"  # Example address

    # Create and initialize the application
    async with create_application() as app:
        try:
            # Process a single wallet
            addresses = [{
                "address": wallet_address,
                "label": "Example Wallet",
                "row_number": 1
            }]

            print(f"📊 Analyzing wallet: {wallet_address[:10]}...{wallet_address[-6:]}")

            results = await app.process_wallet_list(addresses)

            # Display results
            print(f"✅ Analysis completed!")
            print(f"   Total Value: ${results.get('portfolio_values', {}).get('total_usd', 0):,.2f}")
            print(f"   Processing Time: {results.get('performance', {}).get('total_time_seconds', 0):.1f}s")

            return results

        except InvalidAddressError as e:
            print(f"❌ Invalid wallet address: {e}")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise


async def example_2_multiple_wallets():
    """Example 2: Analyze multiple wallet addresses."""

    print("\n🏦 Example 2: Multiple Wallet Analysis")
    print("=" * 50)

    # List of wallet addresses to analyze
    wallet_addresses = [
        "0x742d35Cc6634C0532925a3b8D40e4f337F42090B",
        "0x8ba1f109551bD432803012645Hac136c73F825e01",
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # vitalik.eth
    ]

    async with create_application() as app:
        try:
            # Format addresses for processing
            addresses = []
            for i, addr in enumerate(wallet_addresses):
                addresses.append({
                    "address": addr,
                    "label": f"Wallet {i + 1}",
                    "row_number": i + 1
                })

            print(f"📊 Analyzing {len(addresses)} wallets...")

            results = await app.process_wallet_list(addresses)

            # Display summary
            print(f"✅ Analysis completed!")
            print(f"   Wallets Processed: {results.get('results', {}).get('processed', 0)}")
            print(f"   Total Portfolio Value: ${results.get('portfolio_values', {}).get('total_usd', 0):,.2f}")
            print(f"   Average Value: ${results.get('portfolio_values', {}).get('average_usd', 0):,.2f}")
            print(f"   Active Wallets: {results.get('activity', {}).get('active_wallets', 0)}")

            return results

        except Exception as e:
            print(f"❌ Multiple wallet analysis failed: {e}")
            raise


async def example_3_health_check():
    """Example 3: Check application health before processing."""

    print("\n🏥 Example 3: Health Check")
    print("=" * 30)

    async with create_application() as app:
        try:
            # Perform health check
            health_status = await app.health_check()

            print("Service Health Status:")
            for service, is_healthy in health_status.items():
                status_emoji = "✅" if is_healthy else "❌"
                print(f"  {status_emoji} {service}: {'Healthy' if is_healthy else 'Unhealthy'}")

            # Check if all critical services are healthy
            critical_services = ["ethereum_client", "coingecko_client"]
            all_critical_healthy = all(
                health_status.get(service, False) for service in critical_services
            )

            if all_critical_healthy:
                print("✅ All critical services are healthy - ready for processing!")
                return True
            else:
                print("⚠️  Some critical services are unhealthy - processing may fail")
                return False

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False


async def example_4_configuration_basics():
    """Example 4: Working with configuration."""

    print("\n⚙️ Example 4: Configuration Basics")
    print("=" * 40)

    try:
        # Get current configuration
        config = get_config()

        print("Current Configuration:")
        print(f"  Environment: {config.environment.value}")
        print(f"  Debug Mode: {config.debug}")
        print(f"  Batch Size: {config.processing.batch_size}")
        print(f"  Max Concurrent: {config.processing.max_concurrent_requests}")
        print(f"  Cache Backend: {config.cache.backend.value}")
        print(f"  Ethereum Rate Limit: {config.ethereum.rate_limit}")
        print(f"  CoinGecko Rate Limit: {config.coingecko.rate_limit}")

        # Create custom configuration for testing
        print("\nCustom Configuration Example:")

        # Note: In practice, you'd modify environment variables or config files
        # This is just to show the structure
        print("  You can customize configuration by:")
        print("  1. Setting environment variables (BATCH_SIZE=100)")
        print("  2. Creating a .env file")
        print("  3. Modifying configuration files")

        return config

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise


async def example_5_error_handling():
    """Example 5: Proper error handling patterns."""

    print("\n🛡️ Example 5: Error Handling Patterns")
    print("=" * 45)

    # Example of handling different types of errors
    invalid_addresses = [
        "invalid_address",  # Invalid format
        "0x742d35Cc6634C0532925a3b8D40e4f337F42090B",  # Valid address
        "",  # Empty address
    ]

    async with create_application() as app:
        processed_count = 0
        error_count = 0

        for i, addr in enumerate(invalid_addresses):
            try:
                if not addr.strip():
                    print(f"⏭️  Skipping empty address at position {i}")
                    continue

                addresses = [{
                    "address": addr,
                    "label": f"Test Wallet {i}",
                    "row_number": i
                }]

                print(f"🔍 Testing address: {addr[:20]}...")

                results = await app.process_wallet_list(addresses)

                processed = results.get('results', {}).get('processed', 0)
                if processed > 0:
                    print(f"  ✅ Successfully processed")
                    processed_count += 1
                else:
                    print(f"  ⚠️  Address was skipped or failed")
                    error_count += 1

            except InvalidAddressError:
                print(f"  ❌ Invalid address format")
                error_count += 1

            except EthereumClientError as e:
                print(f"  ❌ Ethereum client error: {e}")
                error_count += 1

            except Exception as e:
                print(f"  ❌ Unexpected error: {e}")
                error_count += 1

        print(f"\nSummary: {processed_count} processed, {error_count} errors")


async def example_6_metrics_collection():
    """Example 6: Collecting and displaying metrics."""

    print("\n📊 Example 6: Metrics Collection")
    print("=" * 40)

    async with create_application() as app:
        try:
            # Collect comprehensive metrics
            metrics = await app.collect_metrics()

            print("Application Metrics:")

            # Display Ethereum client metrics
            if 'ethereum_client' in metrics:
                eth_metrics = metrics['ethereum_client']
                print(f"\n🔗 Ethereum Client:")
                print(f"  Portfolio Requests: {eth_metrics.get('portfolio_requests', 0)}")
                print(f"  Cache Hits: {eth_metrics.get('cache_hits', 0)}")
                print(f"  API Errors: {eth_metrics.get('api_errors', 0)}")
                print(f"  Rate Limit: {eth_metrics.get('rate_limit', 0)} req/min")

            # Display CoinGecko client metrics
            if 'coingecko_client' in metrics:
                cg_metrics = metrics['coingecko_client']
                print(f"\n💰 CoinGecko Client:")
                print(f"  Price Requests: {cg_metrics.get('price_requests', 0)}")
                print(f"  Rate Limit Errors: {cg_metrics.get('rate_limit_errors', 0)}")
                print(f"  Has API Key: {cg_metrics.get('has_api_key', False)}")

            # Display cache metrics
            if 'cache' in metrics:
                cache_metrics = metrics['cache']
                print(f"\n💾 Cache System:")
                for backend, stats in cache_metrics.items():
                    if isinstance(stats, dict):
                        hit_rate = stats.get('hit_rate_percent', 0)
                        print(f"  {backend}: {hit_rate:.1f}% hit rate")

            return metrics

        except Exception as e:
            print(f"❌ Failed to collect metrics: {e}")
            raise


async def example_7_dry_run_mode():
    """Example 7: Using dry run mode for testing."""

    print("\n🧪 Example 7: Dry Run Mode")
    print("=" * 35)

    # Example addresses for testing
    test_addresses = [
        {
            "address": "0x742d35Cc6634C0532925a3b8D40e4f337F42090B",
            "label": "Test Wallet 1",
            "row_number": 1
        },
        {
            "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "label": "Test Wallet 2",
            "row_number": 2
        }
    ]

    async with create_application() as app:
        try:
            print("🧪 Running in dry-run mode (no data will be written)")

            # Process wallets without writing any results
            results = await app.process_wallet_list(test_addresses)

            print("📊 Dry Run Results:")
            print(f"  Wallets that would be processed: {results.get('results', {}).get('processed', 0)}")
            print(f"  Estimated total value: ${results.get('portfolio_values', {}).get('total_usd', 0):,.2f}")
            print(f"  Processing time: {results.get('performance', {}).get('total_time_seconds', 0):.1f}s")
            print("✅ Dry run completed - no actual data was modified")

            return results

        except Exception as e:
            print(f"❌ Dry run failed: {e}")
            raise


async def example_8_working_with_results():
    """Example 8: Working with processing results."""

    print("\n📋 Example 8: Working with Results")
    print("=" * 40)

    async with create_application() as app:
        try:
            # Process some wallets
            addresses = [{
                "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                "label": "Vitalik Buterin",
                "row_number": 1
            }]

            results = await app.process_wallet_list(addresses)

            print("📊 Detailed Results Analysis:")

            # Extract different sections of results
            if 'input' in results:
                input_data = results['input']
                print(f"\n📥 Input Summary:")
                print(f"  Total wallets provided: {input_data.get('total_wallets', 0)}")

            if 'results' in results:
                processing_results = results['results']
                print(f"\n⚙️ Processing Summary:")
                print(f"  Successfully processed: {processing_results.get('processed', 0)}")
                print(f"  Skipped (inactive): {processing_results.get('skipped', 0)}")
                print(f"  Failed: {processing_results.get('failed', 0)}")
                print(f"  Success rate: {processing_results.get('success_rate', 0):.1f}%")

            if 'portfolio_values' in results:
                portfolio = results['portfolio_values']
                print(f"\n💰 Portfolio Analysis:")
                print(f"  Total value: ${portfolio.get('total_usd', 0):,.2f}")
                print(f"  Average value: ${portfolio.get('average_usd', 0):,.2f}")
                print(f"  Median value: ${portfolio.get('median_usd', 0):,.2f}")
                print(f"  Highest wallet: ${portfolio.get('max_usd', 0):,.2f}")
                print(f"  Lowest wallet: ${portfolio.get('min_usd', 0):,.2f}")

            if 'token_holders' in results:
                tokens = results['token_holders']
                print(f"\n🪙 Token Distribution:")
                print(f"  ETH holders: {tokens.get('eth', 0)}")
                print(f"  USDC holders: {tokens.get('usdc', 0)}")
                print(f"  USDT holders: {tokens.get('usdt', 0)}")
                print(f"  DAI holders: {tokens.get('dai', 0)}")

            if 'performance' in results:
                performance = results['performance']
                print(f"\n⚡ Performance Metrics:")
                print(f"  Total processing time: {performance.get('total_time_seconds', 0):.1f}s")
                print(f"  Average time per wallet: {performance.get('average_time_per_wallet', 0):.2f}s")
                print(f"  Cache hit rate: {performance.get('cache_hit_rate', 0):.1f}%")
                print(f"  Total API calls: {performance.get('api_calls_total', 0)}")

            return results

        except Exception as e:
            print(f"❌ Results analysis failed: {e}")
            raise


async def example_9_custom_processing_logic():
    """Example 9: Custom processing with callbacks and monitoring."""

    print("\n🔧 Example 9: Custom Processing Logic")
    print("=" * 45)

    async with create_application() as app:
        try:
            # Example of adding custom monitoring during processing
            addresses = [
                {
                    "address": "0x742d35Cc6634C0532925a3b8D40e4f337F42090B",
                    "label": "Wallet 1",
                    "row_number": 1
                },
                {
                    "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                    "label": "Wallet 2",
                    "row_number": 2
                }
            ]

            print("🔍 Starting custom processing with monitoring...")

            # Before processing - check health
            health_status = await app.health_check()
            healthy_services = sum(1 for is_healthy in health_status.values() if is_healthy)
            total_services = len(health_status)

            print(f"📊 Pre-processing health check: {healthy_services}/{total_services} services healthy")

            # Process with timing
            import time
            start_time = time.time()

            results = await app.process_wallet_list(addresses)

            end_time = time.time()
            processing_duration = end_time - start_time

            # Post-processing analysis
            processed_count = results.get('results', {}).get('processed', 0)
            total_value = results.get('portfolio_values', {}).get('total_usd', 0)

            print(f"⏱️  Custom timing: {processing_duration:.2f}s")
            print(f"📈 Processing rate: {processed_count / processing_duration:.1f} wallets/second")
            print(f"💰 Value per second: ${total_value / processing_duration:,.0f}/second")

            # Check final metrics
            final_metrics = await app.collect_metrics()

            if 'ethereum_client' in final_metrics:
                api_calls = final_metrics['ethereum_client'].get('portfolio_requests', 0)
                print(f"🔗 Total Ethereum API calls: {api_calls}")

            return {
                'results': results,
                'custom_metrics': {
                    'processing_duration': processing_duration,
                    'processing_rate': processed_count / processing_duration if processing_duration > 0 else 0,
                    'value_per_second': total_value / processing_duration if processing_duration > 0 else 0
                }
            }

        except Exception as e:
            print(f"❌ Custom processing failed: {e}")
            raise


async def run_all_examples():
    """Run all basic usage examples."""

    print("🚀 Ethereum Wallet Tracker - Basic Usage Examples")
    print("=" * 60)
    print("This script demonstrates basic usage patterns for the Ethereum Wallet Tracker.")
    print("Make sure you have configured your API keys in the .env file before running.\n")

    examples = [
        example_1_simple_wallet_analysis,
        example_2_multiple_wallets,
        example_3_health_check,
        example_4_configuration_basics,
        example_5_error_handling,
        example_6_metrics_collection,
        example_7_dry_run_mode,
        example_8_working_with_results,
        example_9_custom_processing_logic,
    ]

    results = {}

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 60}")
            print(f"Running Example {i}: {example_func.__name__}")
            print(f"{'=' * 60}")

            result = await example_func()
            results[example_func.__name__] = {
                'status': 'success',
                'result': result
            }

            print(f"✅ Example {i} completed successfully!")

            # Small delay between examples
            await asyncio.sleep(1)

        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            results[example_func.__name__] = {
                'status': 'failed',
                'error': str(e)
            }

            # Continue with other examples even if one fails
            continue

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 EXAMPLES SUMMARY")
    print(f"{'=' * 60}")

    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful

    print(f"Total Examples: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print(f"\nFailed Examples:")
        for name, result in results.items():
            if result['status'] == 'failed':
                print(f"  ❌ {name}: {result['error']}")

    print(f"\n🎉 Basic usage examples completed!")
    print(f"📚 Check the other example files for more advanced usage patterns:")
    print(f"   - examples/advanced_batch.py")
    print(f"   - examples/sheets_integration.py")

    return results


if __name__ == "__main__":
    # Run the examples
    asyncio.run(run_all_examples())