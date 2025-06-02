# Apply conditional formatting for wallet status
status_formatting_rules = [
    ConditionalFormatRule(
        condition_type="TEXT_EQ",
        condition_values=["ACTIVE"],
        background_color=Color(red=0.8, green=1.0, blue=0.8),  # Light green
        font_color=Color(red=0.0, green=0.6, blue=0.0),  # Dark green text
        range_name="G2:G5"
    ),
    ConditionalFormatRule(
        condition_type="TEXT_EQ",
        condition_values=["INACTIVE"],
        background_color=Color(red=0.9, green=0.9, blue=0.9),  # Light gray
        font_color=Color(red=0.5, green=0.5, blue=0.5),  # Gray text
        range_name="G2:G5"
    )
]

for rule in status_formatting_rules:
    await sheets_client.add_conditional_formatting(
        spreadsheet_id=SAMPLE_SPREADSHEET_ID,
        sheet_name=FORMATTED_SHEET_NAME,
        rule=rule
    )

print(f"📊 Applied status-based conditional formatting...")

# Apply data bars for value visualization
value_bars_rule = ConditionalFormatRule(
    condition_type="NUMBER_GREATER",
    condition_values=[0],
    data_bar_color=Color(red=0.2, green=0.6, blue=1.0),  # Blue data bars
    range_name="C2:C5"
)

await sheets_client.add_conditional_formatting(
    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
    sheet_name=FORMATTED_SHEET_NAME,
    rule=value_bars_rule
)

print(f"📊 Added data bars for value visualization...")

# Freeze header row and adjust column widths
await sheets_client.freeze_rows(
    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
    sheet_name=FORMATTED_SHEET_NAME,
    frozen_row_count=1
)

# Auto-resize columns
await sheets_client.auto_resize_columns(
    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
    sheet_name=FORMATTED_SHEET_NAME,
    start_column_index=0,
    end_column_index=7
)

print(f"🔧 Applied sheet layout optimizations...")

# Add summary statistics with charts
summary_data = [
    [""],  # Empty row
    ["PORTFOLIO SUMMARY"],
    ["Total Wallets", len(sample_results) - 1],
    ["Active Wallets", 2],
    ["Inactive Wallets", 2],
    ["Total Portfolio Value", "=SUM(C2:C5)"],
    ["Average Value", "=AVERAGE(C2:C5)"],
    ["Max Value", "=MAX(C2:C5)"],
    ["Min Value", "=MIN(C2:C5)"],
]

await sheets_client.write_range(
    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
    sheet_name=FORMATTED_SHEET_NAME,
    range_name=f"A{len(sample_results) + 1}:B{len(sample_results) + len(summary_data)}",
    values=summary_data
)

# Format summary section
summary_header_style = CellStyle(
    background_color=Color(red=0.1, green=0.3, blue=0.6),
    font_color=Color(red=1.0, green=1.0, blue=1.0),
    bold=True,
    font_size=14
)

await sheets_client.format_range(
    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
    sheet_name=FORMATTED_SHEET_NAME,
    range_name=f"A{len(sample_results) + 2}:B{len(sample_results) + 2}",
    style=summary_header_style
)

print(f"📈 Added formatted summary section...")

print(f"✅ Advanced formatting completed!")
print(f"🎨 Applied formatting includes:")
print(f"   - Header styling with blue background")
print(f"   - Currency formatting for values")
print(f"   - Custom number format for ETH balances")
print(f"   - Conditional formatting for risk levels")
print(f"   - Status-based color coding")
print(f"   - Data bars for value visualization")
print(f"   - Frozen header row")
print(f"   - Auto-resized columns")
print(f"   - Summary statistics with formulas")

return {
    'formatted_data': sample_results,
    'summary_data': summary_data,
    'formatting_applied': True
}

except Exception as e:
print(f"❌ Advanced formatting failed: {e}")
raise


async def example_3_bulk_processing_with_sheets():
    """Example 3: Bulk processing with progress tracking in sheets."""

    print("\n🚀 Example 3: Bulk Processing with Progress Tracking")
    print("=" * 55)

    SAMPLE_SPREADSHEET_ID = "your_spreadsheet_id_here"
    BULK_INPUT_SHEET = "Bulk_Input"
    PROGRESS_SHEET = "Processing_Progress"
    BULK_RESULTS_SHEET = "Bulk_Results"

    async with create_application() as app:
        try:
            sheets_client = app.google_sheets_client

            # Create bulk input data (simulating a large dataset)
            bulk_addresses = []

            # Add some real addresses
            real_addresses = [
                ("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "Vitalik Buterin"),
                ("0x742d35Cc6634C0532925a3b8D40e4f337F42090B", "Example Wallet 1"),
                ("0x8ba1f109551bD432803012645Hac136c73F825e01", "Example Wallet 2"),
            ]

            # Generate test data
            for i, (addr, label) in enumerate(real_addresses):
                bulk_addresses.append([addr, label, f"Real wallet {i + 1}", "HIGH"])

            # Add some test addresses
            for i in range(20):
                fake_addr = f"0x{''.join([f'{j:02x}' for j in range(20)])}"
                bulk_addresses.append([fake_addr, f"Test Wallet {i + 1}", f"Generated for testing", "NORMAL"])

            # Prepare bulk input sheet
            input_headers = ["Address", "Label", "Description", "Priority"]
            input_data = [input_headers] + bulk_addresses

            print(f"📝 Creating bulk input sheet with {len(bulk_addresses)} addresses...")

            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=BULK_INPUT_SHEET,
                range_name=f"A1:D{len(input_data)}",
                values=input_data
            )

            # Setup progress tracking sheet
            progress_headers = [
                "Timestamp", "Phase", "Processed", "Remaining", "Success Rate",
                "Current Value", "Errors", "Est. Completion", "Status"
            ]

            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=PROGRESS_SHEET,
                range_name="A1:I1",
                values=[progress_headers]
            )

            # Format progress sheet header
            header_style = CellStyle(
                background_color=Color(red=0.2, green=0.6, blue=0.2),
                font_color=Color(red=1.0, green=1.0, blue=1.0),
                bold=True
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=PROGRESS_SHEET,
                range_name="A1:I1",
                style=header_style
            )

            print(f"📊 Setup progress tracking sheet...")

            # Configure bulk processing
            bulk_config = BatchConfig(
                batch_size=5,  # Process in small batches for demo
                max_concurrent_jobs_per_batch=2,
                request_delay_seconds=0.5,
                timeout_seconds=120,
                use_cache=True,
                skip_inactive_wallets=True
            )

            # Progress tracking variables
            progress_row = 2
            start_time = datetime.now()
            processed_count = 0
            total_value = Decimal("0")
            errors = 0

            # Progress update function
            async def update_progress(phase: str, status: str = "RUNNING"):
                nonlocal progress_row, processed_count, total_value, errors

                remaining = len(bulk_addresses) - processed_count
                success_rate = (processed_count / max(1, processed_count + errors)) * 100

                # Estimate completion time
                if processed_count > 0:
                    elapsed = datetime.now() - start_time
                    rate = processed_count / elapsed.total_seconds()
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta = datetime.now() + timedelta(seconds=eta_seconds)
                    eta_str = eta.strftime("%H:%M:%S")
                else:
                    eta_str = "Calculating..."

                progress_data = [
                    datetime.now().strftime("%H:%M:%S"),
                    phase,
                    processed_count,
                    remaining,
                    f"{success_rate:.1f}%",
                    f"${float(total_value):,.2f}",
                    errors,
                    eta_str,
                    status
                ]

                await sheets_client.write_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=PROGRESS_SHEET,
                    range_name=f"A{progress_row}:I{progress_row}",
                    values=[progress_data]
                )

                progress_row += 1

            # Start bulk processing with progress updates
            print(f"🚀 Starting bulk processing...")

            await update_progress("INITIALIZATION", "STARTING")

            # Convert addresses for processing
            addresses_for_processing = []
            for i, addr_data in enumerate(bulk_addresses):
                addresses_for_processing.append({
                    "address": addr_data[0],
                    "label": addr_data[1],
                    "row_number": i + 2  # +2 for header row
                })

            await update_progress("VALIDATION", "RUNNING")

            # Process in batches with progress updates
            batch_size = bulk_config.batch_size
            all_results = []

            for i in range(0, len(addresses_for_processing), batch_size):
                batch = addresses_for_processing[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(addresses_for_processing) + batch_size - 1) // batch_size

                await update_progress(f"BATCH_{batch_num}/{total_batches}", "PROCESSING")

                try:
                    # Process this batch
                    batch_results = await app.batch_processor.process_wallet_list(
                        addresses=batch,
                        config_override=bulk_config
                    )

                    # Update progress
                    processed_count += batch_results.wallets_processed
                    total_value += batch_results.total_portfolio_value
                    errors += batch_results.wallets_failed

                    all_results.append(batch_results)

                    await update_progress(f"BATCH_{batch_num}_COMPLETE", "COMPLETED")

                    print(f"📦 Batch {batch_num}/{total_batches} completed: "
                          f"{batch_results.wallets_processed} processed, "
                          f"${batch_results.total_portfolio_value:,.2f} value")

                except Exception as e:
                    errors += len(batch)
                    await update_progress(f"BATCH_{batch_num}_ERROR", "ERROR")
                    print(f"❌ Batch {batch_num} failed: {e}")

                # Small delay between batches
                await asyncio.sleep(1)

            await update_progress("FINALIZING", "COMPLETING")

            # Prepare final results for output sheet
            results_headers = [
                "Address", "Label", "Priority", "Total Value", "ETH Balance",
                "Token Count", "Status", "Processing Time", "Batch Number"
            ]

            results_data = [results_headers]

            # Aggregate all batch results
            total_processed = sum(r.wallets_processed for r in all_results)
            total_portfolio_value = sum(r.total_portfolio_value for r in all_results)

            # Add summary row
            summary_row = [
                f"TOTAL: {total_processed} wallets",
                "",
                "",
                f"${float(total_portfolio_value):,.2f}",
                "",
                "",
                f"{(total_processed / len(bulk_addresses)) * 100:.1f}% success",
                f"{(datetime.now() - start_time).total_seconds():.1f}s",
                f"{len(all_results)} batches"
            ]
            results_data.append(summary_row)

            # Write final results
            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=BULK_RESULTS_SHEET,
                range_name=f"A1:I{len(results_data)}",
                values=results_data
            )

            # Format results sheet
            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=BULK_RESULTS_SHEET,
                range_name="A1:I1",
                style=header_style
            )

            # Format summary row
            summary_style = CellStyle(
                background_color=Color(red=0.9, green=0.9, blue=0.2),
                bold=True
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=BULK_RESULTS_SHEET,
                range_name="A2:I2",
                style=summary_style
            )

            await update_progress("COMPLETE", "FINISHED")

            print(f"✅ Bulk processing completed!")
            print(f"📊 Final Results:")
            print(f"   Total Addresses: {len(bulk_addresses)}")
            print(f"   Successfully Processed: {total_processed}")
            print(f"   Total Portfolio Value: ${float(total_portfolio_value):,.2f}")
            print(f"   Processing Time: {(datetime.now() - start_time).total_seconds():.1f}s")
            print(f"   Success Rate: {(total_processed / len(bulk_addresses)) * 100:.1f}%")

            return {
                'input_count': len(bulk_addresses),
                'processed_count': total_processed,
                'total_value': float(total_portfolio_value),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'batch_results': all_results
            }

        except Exception as e:
            await update_progress("ERROR", "FAILED")
            print(f"❌ Bulk processing failed: {e}")
            raise


async def example_4_multi_sheet_dashboard():
    """Example 4: Create a multi-sheet dashboard with charts and analytics."""

    print("\n📊 Example 4: Multi-Sheet Dashboard Creation")
    print("=" * 50)

    SAMPLE_SPREADSHEET_ID = "your_spreadsheet_id_here"
    DASHBOARD_SHEET = "Dashboard"
    RAW_DATA_SHEET = "Raw_Data"
    ANALYTICS_SHEET = "Analytics"

    async with create_application() as app:
        try:
            sheets_client = app.google_sheets_client

            # Generate sample portfolio data for dashboard
            portfolio_data = [
                ["Address", "Label", "ETH_Balance", "USDC_Balance", "Total_USD", "Risk_Score", "Last_Activity",
                 "Category"],
                ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "Vitalik", 1250.5, 50000, 2500000, 2, "2024-01-15",
                 "Whale"],
                ["0x742d35Cc6634C0532925a3b8D40e4f337F42090B", "DeFi User", 45.2, 15000, 95000, 5, "2024-01-14",
                 "Power User"],
                ["0x8ba1f109551bD432803012645Hac136c73F825e01", "HODLer", 12.8, 5000, 28000, 3, "2024-01-10",
                 "Regular"],
                ["0x123456789abcdef123456789abcdef1234567890", "Trader", 8.5, 25000, 42000, 7, "2024-01-16", "Active"],
                ["0xabcdef123456789abcdef123456789abcdef12345", "Investor", 125.0, 75000, 325000, 4, "2024-01-12",
                 "Whale"],
                ["0x999888777666555444333222111000aaabbbccc", "Small User", 2.1, 1000, 5500, 6, "2023-12-20", "Small"],
            ]

            print(f"📝 Creating raw data sheet...")

            # Write raw data
            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=RAW_DATA_SHEET,
                range_name=f"A1:H{len(portfolio_data)}",
                values=portfolio_data
            )

            # Create dashboard sheet
            dashboard_layout = [
                ["🏦 ETHEREUM WALLET PORTFOLIO DASHBOARD", "", "", "", "", ""],
                [""],
                ["📊 PORTFOLIO OVERVIEW", "", "", "📈 TOP PERFORMERS", "", ""],
                ["Total Wallets", "=COUNTA(Raw_Data!A2:A1000)-1", "", "Wallet", "Value (USD)", ""],
                ["Total Portfolio Value", "=SUM(Raw_Data!E2:E1000)", "",
                 "=INDEX(Raw_Data!B2:B1000,MATCH(MAX(Raw_Data!E2:E1000),Raw_Data!E2:E1000,0))",
                 "=MAX(Raw_Data!E2:E1000)", ""],
                ["Average Balance", "=AVERAGE(Raw_Data!E2:E1000)", "",
                 "=INDEX(Raw_Data!B2:B1000,MATCH(LARGE(Raw_Data!E2:E1000,2),Raw_Data!E2:E1000,0))",
                 "=LARGE(Raw_Data!E2:E1000,2)", ""],
                ["Active Wallets", "=COUNTIF(Raw_Data!G2:G1000,\">\"&(TODAY()-30))", "",
                 "=INDEX(Raw_Data!B2:B1000,MATCH(LARGE(Raw_Data!E2:E1000,3),Raw_Data!E2:E1000,0))",
                 "=LARGE(Raw_Data!E2:E1000,3)", ""],
                [""],
                ["💰 ASSET BREAKDOWN", "", "", "⚠️ RISK ANALYSIS", "", ""],
                ["Total ETH", "=SUM(Raw_Data!C2:C1000)", "", "Low Risk (1-3)", "=COUNTIFS(Raw_Data!F2:F1000,\"<=3\")",
                 ""],
                ["Total USDC", "=SUM(Raw_Data!D2:D1000)", "", "Medium Risk (4-6)",
                 "=COUNTIFS(Raw_Data!F2:F1000,\">=4\",Raw_Data!F2:F1000,\"<=6\")", ""],
                ["ETH Value", "=SUM(Raw_Data!C2:C1000)*2000", "", "High Risk (7-10)",
                 "=COUNTIFS(Raw_Data!F2:F1000,\">=7\")", ""],
                ["USDC Value", "=SUM(Raw_Data!D2:D1000)", "", "Avg Risk Score", "=AVERAGE(Raw_Data!F2:F1000)", ""],
                [""],
                ["📅 ACTIVITY ANALYSIS", "", "", "🏷️ CATEGORY BREAKDOWN", "", ""],
                ["Last 7 Days", "=COUNTIF(Raw_Data!G2:G1000,\">\"&(TODAY()-7))", "", "Whale",
                 "=COUNTIF(Raw_Data!H2:H1000,\"Whale\")", ""],
                ["Last 30 Days", "=COUNTIF(Raw_Data!G2:G1000,\">\"&(TODAY()-30))", "", "Power User",
                 "=COUNTIF(Raw_Data!H2:H1000,\"Power User\")", ""],
                ["Last 90 Days", "=COUNTIF(Raw_Data!G2:G1000,\">\"&(TODAY()-90))", "", "Regular",
                 "=COUNTIF(Raw_Data!H2:H1000,\"Regular\")", ""],
                ["Inactive (>90 days)", "=COUNTIF(Raw_Data!G2:G1000,\"<\"&(TODAY()-90))", "", "Small",
                 "=COUNTIF(Raw_Data!H2:H1000,\"Small\")", ""],
            ]

            print(f"📊 Creating dashboard with formulas...")

            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=DASHBOARD_SHEET,
                range_name=f"A1:F{len(dashboard_layout)}",
                values=dashboard_layout
            )

            # Format dashboard title
            title_style = CellStyle(
                background_color=Color(red=0.1, green=0.2, blue=0.6),
                font_color=Color(red=1.0, green=1.0, blue=1.0),
                bold=True,
                font_size=16
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=DASHBOARD_SHEET,
                range_name="A1:F1",
                style=title_style
            )

            # Format section headers
            section_style = CellStyle(
                background_color=Color(red=0.2, green=0.4, blue=0.8),
                font_color=Color(red=1.0, green=1.0, blue=1.0),
                bold=True,
                font_size=12
            )

            section_ranges = ["A3:C3", "D3:F3", "A9:C9", "D9:F9", "A15:C15", "D15:F15"]
            for range_name in section_ranges:
                await sheets_client.format_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=DASHBOARD_SHEET,
                    range_name=range_name,
                    style=section_style
                )

            # Format currency values
            currency_format = NumberFormat(type="CURRENCY", pattern="$#,##0.00")
            currency_ranges = ["B5:B6", "B12:B13", "E4:E6"]

            for range_name in currency_ranges:
                await sheets_client.format_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=DASHBOARD_SHEET,
                    range_name=range_name,
                    number_format=currency_format
                )

            print(f"🎨 Applied dashboard formatting...")

            # Create analytics sheet with pivot-like analysis
            analytics_data = [
                ["📊 DETAILED ANALYTICS", "", "", "", ""],
                [""],
                ["Category Analysis", "", "", "", ""],
                ["Category", "Count", "Total Value", "Avg Value", "% of Portfolio"],
                ["=UNIQUE(Raw_Data!H2:H1000)", "=COUNTIF(Raw_Data!H2:H1000,A5)",
                 "=SUMIF(Raw_Data!H2:H1000,A5,Raw_Data!E2:E1000)", "=C5/B5", "=C5/SUM(Raw_Data!E2:E1000)"],
                [""],
                ["Risk Distribution", "", "", "", ""],
                ["Risk Level", "Count", "Avg Portfolio Value", "Total at Risk", ""],
                ["Low (1-3)", "=COUNTIFS(Raw_Data!F2:F1000,\"<=3\")",
                 "=AVERAGEIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\"<=3\")",
                 "=SUMIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\"<=3\")", ""],
                ["Medium (4-6)", "=COUNTIFS(Raw_Data!F2:F1000,\">=4\",Raw_Data!F2:F1000,\"<=6\")",
                 "=AVERAGEIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\">=4\",Raw_Data!F2:F1000,\"<=6\")",
                 "=SUMIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\">=4\",Raw_Data!F2:F1000,\"<=6\")", ""],
                ["High (7-10)", "=COUNTIFS(Raw_Data!F2:F1000,\">=7\")",
                 "=AVERAGEIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\">=7\")",
                 "=SUMIFS(Raw_Data!E2:E1000,Raw_Data!F2:F1000,\">=7\")", ""],
                [""],
                ["Performance Metrics", "", "", "", ""],
                ["Metric", "Value", "Benchmark", "Status", ""],
                ["Portfolio Diversity", "=COUNTA(UNIQUE(Raw_Data!H2:H1000))", "5",
                 "=IF(B15>=C15,\"✅ Good\",\"⚠️ Review\")", ""],
                ["Risk Balance", "=AVERAGE(Raw_Data!F2:F1000)", "5", "=IF(B16<=C16,\"✅ Balanced\",\"⚠️ High Risk\")",
                 ""],
                ["Activity Rate", "=COUNTIF(Raw_Data!G2:G1000,\">\"&(TODAY()-30))/COUNTA(Raw_Data!A2:A1000)*100", "70",
                 "=IF(B17>=C17,\"✅ Active\",\"⚠️ Inactive\")", ""],
            ]

            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=ANALYTICS_SHEET,
                range_name=f"A1:E{len(analytics_data)}",
                values=analytics_data
            )

            print(f"📈 Created analytics sheet with formulas...")

            # Format analytics sheet
            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=ANALYTICS_SHEET,
                range_name="A1:E1",
                style=title_style
            )

            # Format section headers in analytics
            analytics_headers = ["A3:E3", "A7:E7", "A13:E13"]
            for range_name in analytics_headers:
                await sheets_client.format_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=ANALYTICS_SHEET,
                    range_name=range_name,
                    style=section_style
                )

            # Add charts (Note: This would require additional Google Sheets API calls for chart creation)
            print(f"📊 Dashboard structure created...")
            print(f"💡 To add charts, use Google Sheets UI to:")
            print(f"   - Create pie chart for category breakdown")
            print(f"   - Create bar chart for risk distribution")
            print(f"   - Create line chart for activity trends")

            print(f"✅ Multi-sheet dashboard created!")
            print(f"📊 Dashboard includes:")
            print(f"   - Portfolio overview with key metrics")
            print(f"   - Top performers analysis")
            print(f"   - Asset breakdown and risk analysis")
            print(f"   - Activity and category analytics")
            print(f"   - Dynamic formulas that update with data")
            print(f"   - Professional formatting and styling")

            return {
                'dashboard_created': True,
                'sheets_created': [DASHBOARD_SHEET, RAW_DATA_SHEET, ANALYTICS_SHEET],
                'data_rows': len(portfolio_data) - 1
            }

        except Exception as e:
            print(f"❌ Dashboard creation failed: {e}")
            raise


async def example_5_automated_reporting():
    """Example 5: Automated reporting with scheduled updates."""

    print("\n🤖 Example 5: Automated Reporting System")
    print("=" * 45)

    SAMPLE_SPREADSHEET_ID = "your_spreadsheet_id_here"
    REPORT_SHEET = "Automated_Report"
    HISTORY_SHEET = "Report_History"

    async with create_application() as app:
        try:
            sheets_client = app.google_sheets_client

            # Generate timestamp for this report
            report_timestamp = datetime.now()
            report_id = report_timestamp.strftime("%Y%m%d_%H%M%S")

            print(f"📝 Generating automated report: {report_id}")

            # Simulate getting portfolio data for multiple time periods
            historical_data = [
                {"date": "2024-01-01", "total_value": 1250000, "active_wallets": 45, "avg_balance": 27777},
                {"date": "2024-01-08", "total_value": 1180000, "active_wallets": 42, "avg_balance": 28095},
                {"date": "2024-01-15", "total_value": 1320000, "active_wallets": 48, "avg_balance": 27500},
                {"date": "2024-01-22", "total_value": 1450000, "active_wallets": 52, "avg_balance": 27884},
            ]

            current_data = {
                "date": report_timestamp.strftime("%Y-%m-%d"),
                "total_value": 1580000,
                "active_wallets": 55,
                "avg_balance": 28727
            }

            # Create automated report
            report_data = [
                ["🤖 AUTOMATED PORTFOLIO REPORT", "", "", "", ""],
                [""],
                [f"Report ID: {report_id}", "", f"Generated: {report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "", ""],
                [""],
                ["📊 CURRENT PORTFOLIO STATUS", "", "", "", ""],
                ["Metric", "Current Value", "Previous Week", "Change", "% Change"],
                ["Total Portfolio Value", f"${current_data['total_value']:,.2f}",
                 f"${historical_data[-1]['total_value']:,.2f}",
                 f"${current_data['total_value'] - historical_data[-1]['total_value']:,.2f}",
                 f"{((current_data['total_value'] - historical_data[-1]['total_value']) / historical_data[-1]['total_value']) * 100:.2f}%"],
                ["Active Wallets", current_data['active_wallets'], historical_data[-1]['active_wallets'],
                 current_data['active_wallets'] - historical_data[-1]['active_wallets'],
                 f"{((current_data['active_wallets'] - historical_data[-1]['active_wallets']) / historical_data[-1]['active_wallets']) * 100:.2f}%"],
                ["Average Balance", f"${current_data['avg_balance']:,.2f}",
                 f"${historical_data[-1]['avg_balance']:,.2f}",
                 f"${current_data['avg_balance'] - historical_data[-1]['avg_balance']:,.2f}",
                 f"{((current_data['avg_balance'] - historical_data[-1]['avg_balance']) / historical_data[-1]['avg_balance']) * 100:.2f}%"],
                [""],
                ["📈 TREND ANALYSIS (4-Week)", "", "", "", ""],
                ["Week", "Portfolio Value", "Active Wallets", "Growth Rate", "Status"],
            ]

            # Add historical trend data
            for i, data in enumerate(historical_data):
                if i == 0:
                    growth_rate = "0.00%"
                    status = "Baseline"
                else:
                    prev_value = historical_data[i - 1]['total_value']
                    growth = ((data['total_value'] - prev_value) / prev_value) * 100
                    growth_rate = f"{growth:.2f}%"
                    status = "📈 Growth" if growth > 0 else "📉 Decline" if growth < 0 else "➡️ Stable"

                report_data.append([
                    f"Week {i + 1}",
                    f"${data['total_value']:,.2f}",
                    data['active_wallets'],
                    growth_rate,
                    status
                ])

            # Add current week
            current_growth = ((current_data['total_value'] - historical_data[-1]['total_value']) / historical_data[-1][
                'total_value']) * 100
            current_status = "📈 Growth" if current_growth > 0 else "📉 Decline" if current_growth < 0 else "➡️ Stable"

            report_data.append([
                "Current Week",
                f"${current_data['total_value']:,.2f}",
                current_data['active_wallets'],
                f"{current_growth:.2f}%",
                current_status
            ])

            # Add insights and recommendations
            report_data.extend([
                [""],
                ["🔍 KEY INSIGHTS", "", "", "", ""],
                ["• Portfolio shows consistent growth trajectory", "", "", "", ""],
                ["• Active wallet count is increasing", "", "", "", ""],
                ["• Average balance per wallet is stable", "", "", "", ""],
                [""],
                ["💡 RECOMMENDATIONS", "", "", "", ""],
                ["• Continue monitoring whale wallet activity", "", "", "", ""],
                ["• Focus on inactive wallet reactivation", "", "", "", ""],
                ["• Implement risk management for large positions", "", "", "", ""],
                [""],
                ["⚠️ ALERTS & NOTIFICATIONS", "", "", "", ""],
            ])

            # Add conditional alerts based on data
            alerts = []

            if current_growth > 10:
                alerts.append("🚨 High growth rate detected - monitor for volatility")
            elif current_growth < -5:
                alerts.append("⚠️ Portfolio decline - investigate cause")

            if current_data['active_wallets'] > historical_data[-1]['active_wallets'] * 1.2:
                alerts.append("📈 Significant increase in active wallets")

            if not alerts:
                alerts.append("✅ No significant alerts - portfolio performing normally")

            for alert in alerts:
                report_data.append([alert, "", "", "", ""])

            # Add footer
            report_data.extend([
                [""],
                ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "", "", "", ""],
                [f"Report generated by Ethereum Wallet Tracker v1.0", "", "", "", ""],
                [f"Next report scheduled: {(report_timestamp + timedelta(days=7)).strftime('%Y-%m-%d')}", "", "", "",
                 ""]
            ])

            # Write report to sheet
            print(f"📊 Writing automated report to sheet...")

            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=REPORT_SHEET,
                range_name=f"A1:E{len(report_data)}",
                values=report_data
            )

            # Apply formatting
            title_style = CellStyle(
                background_color=Color(red=0.1, green=0.3, blue=0.7),
                font_color=Color(red=1.0, green=1.0, blue=1.0),
                bold=True,
                font_size=16
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=REPORT_SHEET,
                range_name="A1:E1",
                style=title_style
            )

            # Format section headers
            section_style = CellStyle(
                background_color=Color(red=0.2, green=0.4, blue=0.8),
                font_color=Color(red=1.0, green=1.0, blue=1.0),
                bold=True
            )

            section_headers = ["A5:E5", "A11:E11", "A18:E18", "A23:E23", "A27:E27"]
            for range_name in section_headers:
                await sheets_client.format_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=REPORT_SHEET,
                    range_name=range_name,
                    style=section_style
                )

            # Format currency columns
            currency_format = NumberFormat(type="CURRENCY", pattern="$#,##0.00")
            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=REPORT_SHEET,
                range_name="B6:D9",
                number_format=currency_format
            )

            # Add conditional formatting for growth indicators
            positive_growth_rule = ConditionalFormatRule(
                condition_type="NUMBER_GREATER",
                condition_values=[0],
                background_color=Color(red=0.8, green=1.0, blue=0.8),
                range_name="D6:D9"
            )

            negative_growth_rule = ConditionalFormatRule(
                condition_type="NUMBER_LESS",
                condition_values=[0],
                background_color=Color(red=1.0, green=0.8, blue=0.8),
                range_name="D6:D9"
            )

            await sheets_client.add_conditional_formatting(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=REPORT_SHEET,
                rule=positive_growth_rule
            )

            await sheets_client.add_conditional_formatting(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=REPORT_SHEET,
                rule=negative_growth_rule
            )

            print(f"🎨 Applied report formatting...")

            # Save report to history
            history_entry = [
                report_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                report_id,
                f"${current_data['total_value']:,.2f}",
                current_data['active_wallets'],
                f"{current_growth:.2f}%",
                current_status.replace("📈 ", "").replace("📉 ", "").replace("➡️ ", ""),
                "✅ Generated Successfully"
            ]

            # Check if history sheet exists and has headers
            try:
                existing_history = await sheets_client.read_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=HISTORY_SHEET,
                    range_name="A1:G1"
                )

                if not existing_history:
                    # Add headers if sheet is empty
                    history_headers = [
                        "Timestamp", "Report ID", "Portfolio Value", "Active Wallets",
                        "Growth Rate", "Trend", "Status"
                    ]
                    await sheets_client.write_range(
                        spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                        sheet_name=HISTORY_SHEET,
                        range_name="A1:G1",
                        values=[history_headers]
                    )

                    # Format headers
                    await sheets_client.format_range(
                        spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                        sheet_name=HISTORY_SHEET,
                        range_name="A1:G1",
                        style=section_style
                    )

                # Find next row for history entry
                history_data = await sheets_client.read_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=HISTORY_SHEET,
                    range_name="A:A"
                )

                next_row = len(history_data) + 1

                await sheets_client.write_range(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=HISTORY_SHEET,
                    range_name=f"A{next_row}:G{next_row}",
                    values=[history_entry]
                )

                print(f"📋 Added report to history log...")

            except Exception as e:
                print(f"⚠️ Failed to update history: {e}")

            # Generate summary statistics
            report_summary = {
                'report_id': report_id,
                'timestamp': report_timestamp.isoformat(),
                'portfolio_value': current_data['total_value'],
                'active_wallets': current_data['active_wallets'],
                'growth_rate': current_growth,
                'trend': current_status,
                'alerts_count': len(alerts),
                'status': 'completed'
            }

            print(f"✅ Automated report completed!")
            print(f"📊 Report Summary:")
            print(f"   Report ID: {report_id}")
            print(f"   Portfolio Value: ${current_data['total_value']:,.2f}")
            print(f"   Growth Rate: {current_growth:.2f}%")
            print(f"   Active Wallets: {current_data['active_wallets']}")
            print(f"   Alerts Generated: {len(alerts)}")
            print(f"   Status: {current_status}")

            return report_summary

        except Exception as e:
            print(f"❌ Automated reporting failed: {e}")
            raise


async def example_6_error_handling_sheets():
    """Example 6: Comprehensive error handling for sheets operations."""

    print("\n🛡️ Example 6: Error Handling for Sheets Operations")
    print("=" * 55)

    async with create_application() as app:
        try:
            sheets_client = app.google_sheets_client

            # Test various error scenarios
            error_scenarios = [
                {
                    "name": "Invalid Spreadsheet ID",
                    "test": lambda: sheets_client.read_range("invalid_id", "Sheet1", "A1:B2"),
                    "expected_error": "SpreadsheetNotFound"
                },
                {
                    "name": "Invalid Range Format",
                    "test": lambda: sheets_client.read_range("valid_id", "Sheet1", "INVALID_RANGE"),
                    "expected_error": "InvalidRange"
                },
                {
                    "name": "Non-existent Sheet",
                    "test": lambda: sheets_client.read_range("valid_id", "NonExistentSheet", "A1:B2"),
                    "expected_error": "SheetNotFound"
                },
                {
                    "name": "Permission Denied",
                    "test": lambda: sheets_client.write_range("no_permission_id", "Sheet1", "A1:B2", [["test"]]),
                    "expected_error": "PermissionDenied"
                }
            ]

            error_results = []

            print(f"🧪 Testing error handling scenarios...")

            for scenario in error_scenarios:
                print(f"\n🔍 Testing: {scenario['name']}")

                try:
                    # Attempt the operation
                    await scenario['test']()

                    # If we get here, the operation unexpectedly succeeded
                    result = {
                        'scenario': scenario['name'],
                        'status': 'unexpected_success',
                        'error': None
                    }
                    print(f"  ⚠️ Unexpected success - operation should have failed")

                except GoogleSheetsClientError as e:
                    # Expected Google Sheets error
                    result = {
                        'scenario': scenario['name'],
                        'status': 'expected_error',
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                    print(f"  ✅ Caught expected error: {type(e).__name__}")

                except Exception as e:
                    # Unexpected error type
                    result = {
                        'scenario': scenario['name'],
                        'status': 'unexpected_error',
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                    print(f"  ❌ Unexpected error type: {type(e).__name__}")

                error_results.append(result)

            # Test recovery strategies
            print(f"\n🔧 Testing Error Recovery Strategies...")

            recovery_tests = [
                {
                    "name": "Retry on Timeout",
                    "strategy": "exponential_backoff"
                },
                {
                    "name": "Fallback to Read-only Mode",
                    "strategy": "graceful_degradation"
                },
                {
                    "name": "Partial Success Handling",
                    "strategy": "partial_processing"
                }
            ]

            for test in recovery_tests:
                print(f"\n🔄 Testing Recovery: {test['name']}")

                if test['strategy'] == 'exponential_backoff':
                    # Simulate retry with exponential backoff
                    max_retries = 3
                    base_delay = 1.0

                    for attempt in range(max_retries):
                        try:
                            print(f"  Attempt {attempt + 1}/{max_retries}")

                            # Simulate operation that might fail
                            if attempt < 2:  # Fail first two attempts
                                raise GoogleSheetsClientError("Rate limit exceeded")
                            else:
                                print(f"  ✅ Operation succeeded on attempt {attempt + 1}")
                                break

                        except GoogleSheetsClientError as e:
                            if attempt == max_retries - 1:
                                print(f"  ❌ All retry attempts failed: {e}")
                            else:
                                delay = base_delay * (2 ** attempt)
                                print(f"  ⏳ Retrying in {delay} seconds...")
                                await asyncio.sleep(delay)

                elif test['strategy'] == 'graceful_degradation':
                    # Test fallback to read-only mode
                    try:
                        # Attempt write operation
                        print(f"  Attempting write operation...")
                        raise GoogleSheetsClientError("Write permission denied")

                    except GoogleSheetsClientError:
                        print(f"  ⚠️ Write failed, falling back to read-only mode")
                        print(f"  ✅ Successfully switched to read-only operations")

                elif test['strategy'] == 'partial_processing':
                    # Test handling partial success in batch operations
                    batch_items = ["item1", "item2", "item3", "item4", "item5"]
                    successful_items = []
                    failed_items = []

                    for i, item in enumerate(batch_items):
                        try:
                            # Simulate that some items fail
                            if i in [1, 3]:  # Fail items 2 and 4
                                raise GoogleSheetsClientError(f"Failed to process {item}")

                            successful_items.append(item)
                            print(f"  ✅ Processed: {item}")

                        except GoogleSheetsClientError as e:
                            failed_items.append(item)
                            print(f"  ❌ Failed: {item}")

                    print(f"  📊 Batch results: {len(successful_items)} success, {len(failed_items)} failed")
                    print(f"  ✅ Partial success handling completed")

            # Test input validation and sanitization
            print(f"\n🛡️ Testing Input Validation...")

            validation_tests = [
                {
                    "name": "Range Validation",
                    "input": "A1:XYZ999999",
                    "valid": False
                },
                {
                    "name": "Sheet Name Validation",
                    "input": "Sheet/With\\Invalid:Characters",
                    "valid": False
                },
                {
                    "name": "Data Size Validation",
                    "input": [["x"] * 1000] * 1000,  # Very large dataset
                    "valid": False
                },
                {
                    "name": "Valid Range",
                    "input": "A1:Z100",
                    "valid": True
                }
            ]

            for test in validation_tests:
                print(f"\n🔍 Validating: {test['name']}")

                try:
                    # Simulate validation logic
                    if test['name'] == "Range Validation":
                        # Check if range format is valid
                        range_input = test['input']
                        if "XYZ" in range_input or len(range_input) > 20:
                            raise ValueError("Invalid range format")

                    elif test['name'] == "Sheet Name Validation":
                        # Check for invalid characters
                        sheet_name = test['input']
                        invalid_chars = ['/', '\\', ':', '*', '?', '[', ']']
                        if any(char in sheet_name for char in invalid_chars):
                            raise ValueError("Invalid characters in sheet name")

                    elif test['name'] == "Data Size Validation":
                        # Check data size limits
                        data = test['input']
                        if len(data) > 500 or (len(data) > 0 and len(data[0]) > 500):
                            raise ValueError("Data exceeds size limits")

                    if test['valid']:
                        print(f"  ✅ Validation passed as expected")
                    else:
                        print(f"  ⚠️ Validation should have failed but passed")

                except ValueError as e:
                    if not test['valid']:
                        print(f"  ✅ Validation correctly rejected input: {e}")
                    else:
                        print(f"  ❌ Validation incorrectly rejected valid input: {e}")

            # Generate error handling report
            print(f"\n📊 Error Handling Test Results:")

            expected_errors = sum(1 for r in error_results if r['status'] == 'expected_error')
            unexpected_errors = sum(1 for r in error_results if r['status'] == 'unexpected_error')
            unexpected_successes = sum(1 for r in error_results if r['status'] == 'unexpected_success')

            print(f"  Error Scenario Tests: {len(error_results)}")
            print(f"    ✅ Expected Errors: {expected_errors}")
            print(f"    ❌ Unexpected Errors: {unexpected_errors}")
            print(f"    ⚠️ Unexpected Successes: {unexpected_successes}")

            print(f"  Recovery Strategy Tests: {len(recovery_tests)}")
            print(f"    ✅ All recovery strategies tested")

            print(f"  Input Validation Tests: {len(validation_tests)}")
            print(f"    ✅ All validation scenarios tested")

            # Best practices summary
            print(f"\n💡 Error Handling Best Practices Demonstrated:")
            print(f"   🔄 Exponential backoff for rate limiting")
            print(f"   🛡️ Graceful degradation for permission issues")
            print(f"   📊 Partial success handling for batch operations")
            print(f"   ✅ Input validation and sanitization")
            print(f"   📋 Comprehensive error logging and reporting")
            print(f"   🔧 Recovery strategy implementation")

            return {
                'error_scenarios_tested': len(error_scenarios),
                'recovery_strategies_tested': len(recovery_tests),
                'validation_tests': len(validation_tests),
                'expected_errors': expected_errors,
                'unexpected_issues': unexpected_errors + unexpected_successes,
                'error_results': error_results
            }

        except Exception as e:
            print(f"❌ Error handling testing failed: {e}")
            raise


async def run_all_sheets_examples():
    """Run all Google Sheets integration examples."""

    print("📊 Ethereum Wallet Tracker - Google Sheets Integration Examples")
    print("=" * 70)
    print("This script demonstrates comprehensive Google Sheets integration")
    print("including data formatting, dashboard creation, and error handling.")
    print("\n⚠️ IMPORTANT: Before running these examples:")
    print("   1. Set up Google Sheets API credentials")
    print("   2. Update SAMPLE_SPREADSHEET_ID with your actual sheet ID")
    print("   3. Ensure the service account has access to your spreadsheet")
    print("   4. Grant edit permissions for full functionality\n")

    examples = [
        example_1_basic_sheets_integration,
        example_2_advanced_formatting,
        example_3_bulk_processing_with_sheets,
        example_4_multi_sheet_dashboard,
        example_5_automated_reporting,
        example_6_error_handling_sheets,
    ]

    results = {}

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 70}")
            print(f"Running Sheets Example {i}: {example_func.__name__}")
            print(f"{'=' * 70}")

            result = await example_func()
            results[example_func.__name__] = {
                'status': 'success',
                'result': result
            }

            print(f"✅ Sheets Example {i} completed successfully!")

            # Delay between examples to avoid rate limiting
            await asyncio.sleep(2)

        except Exception as e:
            print(f"❌ Sheets Example {i} failed: {e}")
            results[example_func.__name__] = {
                'status': 'failed',
                'error': str(e)
            }

            # Continue with other examples even if one fails
            continue

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 GOOGLE SHEETS INTEGRATION SUMMARY")
    print(f"{'=' * 70}")

    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = len(results) - successful

    print(f"Total Sheets Examples: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print(f"\nFailed Examples:")
        for name, result in results.items():
            if result['status'] == 'failed':
                print(f"  ❌ {name}: {result['error']}")

    if successful > 0:
        print(f"\n📈 Google Sheets Capabilities Demonstrated:")
        print(f"   📊 Basic data reading and writing")
        print(f"   🎨 Advanced formatting and styling")
        print(f"   📈 Dashboard creation with charts")
        print(f"   🤖 Automated reporting systems")
        print(f"   🚀 Bulk processing with progress tracking")
        print(f"   🛡️ Comprehensive error handling")

        print(f"\n🎯 Key Features Showcased:")
        print(f"   • Conditional formatting based on data values")
        print(f"   • Currency and number formatting")
        print(f"   • Multi-sheet workbook management")
        print(f"   • Real-time progress tracking")
        print(f"   • Dynamic formulas and calculations")
        print(f"   • Professional report generation")
        print(f"   • Robust error recovery strategies")

    print(f"\n💡 Next Steps for Production Use:")
    print(f"   📋 Set up proper Google Cloud project and credentials")
    print(f"   🔐 Implement OAuth 2.0 for user authentication")
    print(f"   📊 Create custom dashboard templates")
    print(f"   ⏰ Set up scheduled report generation")
    print(f"   🔔 Implement notification systems")
    print(f"   📈 Add more advanced charting and visualization")

    print(f"\n🎉 Google Sheets integration examples completed!")

    return results


if __name__ == "__main__":
    # Run the Google Sheets integration examples
    asyncio.run(run_all_sheets_examples())
"""Google Sheets integration examples for the Ethereum Wallet Tracker.

This file demonstrates:
- Reading wallet addresses from Google Sheets
- Writing results back to sheets with formatting
- Different input/output formats
- Advanced styling and conditional formatting
- Batch processing with sheets integration
- Error handling for sheets operations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any, Optional

from wallet_tracker.app import Application, create_application
from wallet_tracker.config import get_config
from wallet_tracker.clients.google_sheets_client import (
    GoogleSheetsClient,
    GoogleSheetsClientError,
    SheetFormatting,
    ConditionalFormatRule,
    CellStyle,
    Color,
    NumberFormat
)
from wallet_tracker.processors.batch_types import BatchConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_basic_sheets_integration():
    """Example 1: Basic Google Sheets integration with wallet analysis."""

    print("📊 Example 1: Basic Google Sheets Integration")
    print("=" * 50)

    # Note: You'll need to update these with your actual Google Sheets details
    SAMPLE_SPREADSHEET_ID = "your_spreadsheet_id_here"
    INPUT_SHEET_NAME = "Wallet_Addresses"
    OUTPUT_SHEET_NAME = "Analysis_Results"

    async with create_application() as app:
        try:
            # Initialize Google Sheets client
            sheets_client = app.google_sheets_client

            print(f"🔗 Connecting to Google Sheets...")

            # Test connection
            try:
                sheet_info = await sheets_client.get_spreadsheet_info(SAMPLE_SPREADSHEET_ID)
                print(f"✅ Connected to: {sheet_info.get('properties', {}).get('title', 'Unknown')}")
            except GoogleSheetsClientError as e:
                print(f"❌ Failed to connect to Google Sheets: {e}")
                print(f"💡 Make sure you have:")
                print(f"   - Valid Google Sheets API credentials")
                print(f"   - Access to the specified spreadsheet")
                print(f"   - Updated SAMPLE_SPREADSHEET_ID with your actual sheet ID")
                return None

            # Create sample input data in sheets format
            sample_wallet_data = [
                ["Address", "Label", "Notes"],
                ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "Vitalik Buterin", "Ethereum co-founder"],
                ["0x742d35Cc6634C0532925a3b8D40e4f337F42090B", "Example Wallet", "Test wallet"],
                ["0x8ba1f109551bD432803012645Hac136c73F825e01", "Another Wallet", "Sample data"],
            ]

            # Write sample data to input sheet
            print(f"📝 Writing sample data to input sheet...")
            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=INPUT_SHEET_NAME,
                range_name="A1:C4",
                values=sample_wallet_data
            )

            # Read wallet addresses from the sheet
            print(f"📖 Reading wallet addresses from sheet...")
            input_data = await sheets_client.read_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=INPUT_SHEET_NAME,
                range_name="A:C"
            )

            # Process the input data
            addresses = []
            if input_data and len(input_data) > 1:  # Skip header row
                for i, row in enumerate(input_data[1:], 1):
                    if len(row) >= 1 and row[0].strip():  # Has address
                        addresses.append({
                            "address": row[0].strip(),
                            "label": row[1].strip() if len(row) > 1 else f"Wallet {i}",
                            "row_number": i + 1  # +1 for header row
                        })

            print(f"📊 Found {len(addresses)} wallet addresses to analyze")

            if not addresses:
                print(f"⚠️  No valid wallet addresses found in the sheet")
                return None

            # Process the wallets
            print(f"⚡ Processing wallets...")
            results = await app.process_wallet_list(addresses)

            # Prepare results for writing back to sheets
            output_headers = [
                "Row", "Address", "Label", "Total Value (USD)", "ETH Balance",
                "Token Count", "Last Activity", "Status", "Processing Time"
            ]

            output_data = [output_headers]

            # Add results data
            wallet_results = results.get('wallet_details', [])
            for wallet in wallet_results:
                output_data.append([
                    wallet.get('row_number', ''),
                    wallet.get('address', ''),
                    wallet.get('label', ''),
                    f"${wallet.get('total_value_usd', 0):,.2f}",
                    f"{wallet.get('eth_balance', 0):.4f}",
                    wallet.get('token_count', 0),
                    wallet.get('last_activity', 'Unknown'),
                    wallet.get('status', 'Unknown'),
                    f"{wallet.get('processing_time', 0):.2f}s"
                ])

            # Add summary row
            summary_row = [
                "SUMMARY",
                f"{results.get('results', {}).get('processed', 0)} wallets",
                "",
                f"${results.get('portfolio_values', {}).get('total_usd', 0):,.2f}",
                "",
                "",
                "",
                f"{results.get('results', {}).get('success_rate', 0):.1f}% success",
                f"{results.get('performance', {}).get('total_time_seconds', 0):.1f}s"
            ]
            output_data.append(summary_row)

            # Write results to output sheet
            print(f"📝 Writing results to output sheet...")
            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=OUTPUT_SHEET_NAME,
                range_name=f"A1:I{len(output_data)}",
                values=output_data
            )

            print(f"✅ Results written to Google Sheets!")
            print(f"📊 Processing Summary:")
            print(f"  Wallets Processed: {results.get('results', {}).get('processed', 0)}")
            print(f"  Total Value: ${results.get('portfolio_values', {}).get('total_usd', 0):,.2f}")
            print(f"  Sheet URL: https://docs.google.com/spreadsheets/d/{SAMPLE_SPREADSHEET_ID}")

            return {
                'input_data': input_data,
                'processing_results': results,
                'output_data': output_data
            }

        except Exception as e:
            print(f"❌ Basic sheets integration failed: {e}")
            raise


async def example_2_advanced_formatting():
    """Example 2: Advanced formatting and styling of results."""

    print("\n🎨 Example 2: Advanced Formatting & Styling")
    print("=" * 45)

    SAMPLE_SPREADSHEET_ID = "your_spreadsheet_id_here"
    FORMATTED_SHEET_NAME = "Formatted_Results"

    async with create_application() as app:
        try:
            sheets_client = app.google_sheets_client

            # Sample processed data for formatting demonstration
            sample_results = [
                ["Address", "Label", "Total Value", "ETH Balance", "Risk Level", "Last Activity", "Status"],
                ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "Vitalik", 1500000.50, 1234.5678, "LOW", "2024-01-15",
                 "ACTIVE"],
                ["0x742d35Cc6634C0532925a3b8D40e4f337F42090B", "Whale Wallet", 850000.25, 890.1234, "MEDIUM",
                 "2024-01-10", "ACTIVE"],
                ["0x8ba1f109551bD432803012645Hac136c73F825e01", "Regular User", 45000.75, 12.3456, "LOW", "2023-12-20",
                 "INACTIVE"],
                ["0x123456789abcdef123456789abcdef1234567890", "Small Holder", 150.00, 0.0789, "HIGH", "2023-10-15",
                 "INACTIVE"],
            ]

            print(f"📝 Writing sample data with advanced formatting...")

            # Write the data first
            await sheets_client.write_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=FORMATTED_SHEET_NAME,
                range_name=f"A1:G{len(sample_results)}",
                values=sample_results
            )

            # Apply header formatting
            header_style = CellStyle(
                background_color=Color(red=0.2, green=0.4, blue=0.8),  # Blue background
                font_color=Color(red=1.0, green=1.0, blue=1.0),  # White text
                bold=True,
                font_size=12
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=FORMATTED_SHEET_NAME,
                range_name="A1:G1",
                style=header_style
            )

            print(f"🎨 Applied header formatting...")

            # Format currency columns
            currency_format = NumberFormat(
                type="CURRENCY",
                pattern="$#,##0.00"
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=FORMATTED_SHEET_NAME,
                range_name="C2:C5",  # Total Value column
                number_format=currency_format
            )

            # Format ETH balance with custom number format
            eth_format = NumberFormat(
                type="NUMBER",
                pattern="#,##0.0000\" ETH\""
            )

            await sheets_client.format_range(
                spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                sheet_name=FORMATTED_SHEET_NAME,
                range_name="D2:D5",  # ETH Balance column
                number_format=eth_format
            )

            print(f"💰 Applied currency and number formatting...")

            # Apply conditional formatting for risk levels
            risk_formatting_rules = [
                ConditionalFormatRule(
                    condition_type="TEXT_EQ",
                    condition_values=["LOW"],
                    background_color=Color(red=0.8, green=1.0, blue=0.8),  # Light green
                    range_name="E2:E5"
                ),
                ConditionalFormatRule(
                    condition_type="TEXT_EQ",
                    condition_values=["MEDIUM"],
                    background_color=Color(red=1.0, green=1.0, blue=0.8),  # Light yellow
                    range_name="E2:E5"
                ),
                ConditionalFormatRule(
                    condition_type="TEXT_EQ",
                    condition_values=["HIGH"],
                    background_color=Color(red=1.0, green=0.8, blue=0.8),  # Light red
                    range_name="E2:E5"
                )
            ]

            for rule in risk_formatting_rules:
                await sheets_client.add_conditional_formatting(
                    spreadsheet_id=SAMPLE_SPREADSHEET_ID,
                    sheet_name=FORMATTED_SHEET_NAME,
                    rule=rule
                )

            print(f"🚦 Applied conditional formatting for risk levels...")

            # Apply conditional formatting for wallet