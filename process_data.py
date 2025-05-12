import pandas as pd
import json
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def convert_value(value):
    """Helper function to convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(value)
    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(value):
            return None
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif pd.isna(value):
        return None
    return value

def process_sales_data():
    """
    ประมวลผลข้อมูลการขายและสต็อกสินค้าจากไฟล์ CSV ที่ระบุ
    และสร้างข้อมูลสรุปในรูปแบบ JSON สำหรับ Dashboard
    ปรับปรุง Branch Analysis ให้สามารถ filter รายเดือนได้
    """

    sales_files_info = {
        'EW-ALL-SALES-DATA - 25JAN.csv': '2025-01',
        'EW-ALL-SALES-DATA - 25FEB.csv': '2025-02',
        'EW-ALL-SALES-DATA - 25MAR.csv': '2025-03',
        'EW-ALL-SALES-DATA - 25APR.csv': '2025-04',
    }

    stock_files_info = {
        'stock-2501.csv': '2025-01',
        'stock-2502.csv': '2025-02',
        'stock-2503.csv': '2025-03',
        'Stock-2504.csv': '2025-04',
        'Stock-2505.csv': '2025-05',
    }

    sales_col_mapping = {
        'SKU': 'item_code', 'Product Name': 'item_name', 'Branch': 'branch',
        'Qty': 'quantity', 'Value': 'revenue'
    }
    required_sales_cols_renamed = ['item_code', 'item_name', 'branch', 'quantity', 'revenue']
    stock_col_mapping = { 'SKU': 'item_code', 'Name': 'item_name', 'Total': 'on_hand' }
    required_stock_cols_renamed = ['item_code', 'item_name', 'on_hand']

    all_sales_data = []
    for filename, month_year_str in sales_files_info.items():
        try:
            df = pd.read_csv(filename, encoding='utf-8')
            missing_original_sales_cols = [orig_col for orig_col in sales_col_mapping.keys() if orig_col not in df.columns]
            if missing_original_sales_cols: print(f"คำเตือน: ไฟล์ขาย {filename} ขาดคอลัมน์: {', '.join(missing_original_sales_cols)}. ข้ามไฟล์."); continue
            df.rename(columns=sales_col_mapping, inplace=True)
            if not all(col in df.columns for col in required_sales_cols_renamed): print(f"คำเตือน: ไฟล์ขาย {filename} หลัง rename ขาดคอลัมน์. ข้ามไฟล์."); continue
            df['month_year'] = month_year_str
            df['branch'] = df['branch'].astype(str).str.strip().fillna('Unknown')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
            all_sales_data.append(df[required_sales_cols_renamed + ['month_year']])
        except FileNotFoundError: print(f"คำเตือน: ไม่พบไฟล์ขาย {filename}")
        except UnicodeDecodeError:
            try: 
                df = pd.read_csv(filename, encoding='iso-8859-11')
                missing_original_sales_cols = [orig_col for orig_col in sales_col_mapping.keys() if orig_col not in df.columns]; 
                if missing_original_sales_cols: continue
                df.rename(columns=sales_col_mapping, inplace=True)
                if not all(col in df.columns for col in required_sales_cols_renamed): continue
                df['month_year'] = month_year_str; df['branch'] = df['branch'].astype(str).str.strip().fillna('Unknown')
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0); df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
                all_sales_data.append(df[required_sales_cols_renamed + ['month_year']])
                print(f"ข้อมูล: อ่านไฟล์ขาย {filename} ด้วย 'iso-8859-11'")
            except Exception as e_fallback: print(f"คำเตือน: ไม่สามารถอ่านไฟล์ขาย {filename} ด้วย 'utf-8' หรือ 'iso-8859-11'. {e_fallback}. ข้ามไฟล์.")
        except Exception as e: print(f"เกิดข้อผิดพลาดในการอ่านไฟล์ขาย {filename}: {e}")

    if not all_sales_data: print("ข้อผิดพลาด: ไม่สามารถโหลดข้อมูลการขายได้เลย"); return None
    sales_df = pd.concat(all_sales_data, ignore_index=True)
    sales_df['item_code'] = sales_df['item_code'].astype(str).str.strip()
    sales_df['item_name'] = sales_df['item_name'].astype(str).str.strip()

    all_stock_data = [] # (Stock loading logic - assuming it's correct from previous versions)
    for filename, month_year_str in stock_files_info.items():
        try:
            df_stock = pd.read_csv(filename, encoding='utf-8')
            missing_original_stock_cols = [orig_col for orig_col in stock_col_mapping.keys() if orig_col not in df_stock.columns]; 
            if missing_original_stock_cols: continue
            df_stock.rename(columns=stock_col_mapping, inplace=True)
            if not all(col in df_stock.columns for col in required_stock_cols_renamed): continue
            df_stock['month_year'] = month_year_str; df_stock['on_hand'] = pd.to_numeric(df_stock['on_hand'], errors='coerce').fillna(0)
            all_stock_data.append(df_stock[required_stock_cols_renamed + ['month_year']])
        except FileNotFoundError: print(f"คำเตือน: ไม่พบไฟล์สต็อก {filename}")
        except UnicodeDecodeError:
            try: 
                df_stock = pd.read_csv(filename, encoding='iso-8859-11')
                missing_original_stock_cols = [orig_col for orig_col in stock_col_mapping.keys() if orig_col not in df_stock.columns];
                if missing_original_stock_cols: continue
                df_stock.rename(columns=stock_col_mapping, inplace=True)
                if not all(col in df_stock.columns for col in required_stock_cols_renamed): continue
                df_stock['month_year'] = month_year_str; df_stock['on_hand'] = pd.to_numeric(df_stock['on_hand'], errors='coerce').fillna(0)
                all_stock_data.append(df_stock[required_stock_cols_renamed + ['month_year']])
                print(f"ข้อมูล: อ่านไฟล์สต็อก {filename} ด้วย 'iso-8859-11'")
            except Exception as e_fallback: print(f"คำเตือน: ไม่สามารถอ่านไฟล์สต็อก {filename} ด้วย 'utf-8' หรือ 'iso-8859-11'. {e_fallback}. ข้ามไฟล์.")
        except Exception as e: print(f"เกิดข้อผิดพลาดในการอ่านไฟล์สต็อก {filename}: {e}")

    stock_df = pd.concat(all_stock_data, ignore_index=True) if all_stock_data else pd.DataFrame(columns=required_stock_cols_renamed + ['month_year'])
    if not stock_df.empty:
        stock_df['item_code'] = stock_df['item_code'].astype(str).str.strip()
        stock_df['item_name'] = stock_df['item_name'].astype(str).str.strip()

    monthly_sales_agg_for_products = sales_df.groupby(['month_year', 'item_code', 'item_name']).agg(total_quantity_sold=('quantity', 'sum'), total_revenue=('revenue', 'sum')).reset_index()
    unique_item_codes_in_sales = sales_df['item_code'].unique()
    product_codes = sorted(unique_item_codes_in_sales)
    active_months = sorted(sales_df['month_year'].unique()) 
    
    idx = pd.MultiIndex.from_product([product_codes, active_months], names=['item_code', 'month_year'])
    master_df = pd.DataFrame(index=idx).reset_index()
    master_df = pd.merge(master_df, monthly_sales_agg_for_products, on=['item_code', 'month_year'], how='left')
    item_name_map_sales = sales_df[['item_code', 'item_name']].drop_duplicates().set_index('item_code')['item_name']
    item_name_map_stock = stock_df[['item_code', 'item_name']].drop_duplicates().set_index('item_code')['item_name']
    item_name_map = item_name_map_stock.combine_first(item_name_map_sales)
    master_df['item_name'] = master_df['item_code'].map(item_name_map)
    if 'item_name_x' in master_df.columns: master_df['item_name'] = master_df['item_name_x'].fillna(master_df['item_code'].map(item_name_map)); master_df.drop(columns=['item_name_x', 'item_name_y'], inplace=True, errors='ignore')
    master_df['item_name'] = master_df['item_name'].fillna('N/A')
    master_df = pd.merge(master_df, stock_df[['item_code', 'month_year', 'on_hand']], on=['item_code', 'month_year'], how='left') if not stock_df.empty else master_df.assign(on_hand=0.0)
    master_df.rename(columns={'on_hand': 'bom_stock'}, inplace=True)
    if not stock_df.empty:
        stock_df_shifted = stock_df.copy(); stock_df_shifted['month_year_dt'] = pd.to_datetime(stock_df_shifted['month_year'] + '-01', errors='coerce'); stock_df_shifted = stock_df_shifted.dropna(subset=['month_year_dt'])
        stock_df_shifted['prev_month_year'] = (stock_df_shifted['month_year_dt'] - pd.DateOffset(months=1)).dt.strftime('%Y-%m')
        master_df = pd.merge(master_df, stock_df_shifted[['item_code', 'prev_month_year', 'on_hand']], left_on=['item_code', 'month_year'], right_on=['item_code', 'prev_month_year'], how='left')
        master_df.rename(columns={'on_hand': 'eom_stock'}, inplace=True); master_df.drop(columns=['prev_month_year'], inplace=True, errors='ignore')
    else: master_df['eom_stock'] = 0.0
    master_df[['total_quantity_sold', 'total_revenue', 'bom_stock', 'eom_stock']] = master_df[['total_quantity_sold', 'total_revenue', 'bom_stock', 'eom_stock']].fillna(0.0)

    products_data_list = [] # (Product analysis logic - assuming correct)
    overall_sales_by_product = master_df.groupby(['item_code', 'item_name']).agg(overall_quantity=('total_quantity_sold', 'sum'), overall_revenue=('total_revenue', 'sum'), avg_bom_stock=('bom_stock', lambda x: convert_value(x[x > 0].mean()) if (x > 0).any() else 0)).reset_index()
    best_sellers_overall_df = overall_sales_by_product.sort_values(by='overall_revenue', ascending=False).head(10)
    num_months_data = len(active_months) if active_months else 1; overall_sales_by_product['avg_monthly_sales'] = overall_sales_by_product['overall_quantity'] / num_months_data
    slow_movers_overall_df = overall_sales_by_product[(overall_sales_by_product['avg_monthly_sales'] < 10) & (overall_sales_by_product['avg_bom_stock'] > 0)].sort_values(by='avg_monthly_sales', ascending=True).head(10)
    potential_stock_issues = [] 
    for item_code_val in product_codes: # (Detailed product analysis loop - assuming correct)
        product_df_item = master_df[master_df['item_code'] == item_code_val].sort_values(by='month_year'); item_name_val = product_df_item['item_name'].iloc[0] if not product_df_item.empty and product_df_item['item_name'].iloc[0] != 'N/A' else item_code_val; monthly_data_for_product = []
        product_df_item['prev_month_quantity_sold'] = product_df_item['total_quantity_sold'].shift(1); product_df_item['prev_month_eom_stock'] = product_df_item['eom_stock'].shift(1)
        for _, row in product_df_item.iterrows():
            analysis_text, current_sales, prev_sales, bom_stock, eom_stock, prev_eom_stock = "ข้อมูลไม่เพียงพอ", row['total_quantity_sold'], row['prev_month_quantity_sold'], row['bom_stock'], row['eom_stock'], row['prev_month_eom_stock']; issue_key = f"{item_code_val}_{row['month_year']}"
            if pd.isna(prev_sales): analysis_text = "เดือนแรกของข้อมูลสินค้า"; # ... (rest of analysis logic)
            elif current_sales > prev_sales: analysis_text = "ยอดขายเพิ่มขึ้น"; # ...
            elif current_sales < prev_sales: analysis_text = "ยอดขายลดลง"; # ...
            elif current_sales == 0: analysis_text = "ยอดขายเป็นศูนย์"; # ...
            elif current_sales == prev_sales: analysis_text = "ยอดขายคงที่"; # ...
            # Simplified stock issue logging for brevity
            if bom_stock == 0 and current_sales == 0 and not any(p['issueKey'] == issue_key for p in potential_stock_issues): potential_stock_issues.append({"itemCode": item_code_val, "itemName": item_name_val, "monthYear": row['month_year'], "issue": "สินค้าหมดสต็อก (BOM=0) และไม่มีการขาย", "issueKey": issue_key})
            if bom_stock > 0 and eom_stock == 0 and current_sales > 0 and not any(p['issueKey'] == issue_key for p in potential_stock_issues): potential_stock_issues.append({"itemCode": item_code_val, "itemName": item_name_val, "monthYear": row['month_year'], "issue": "สินค้าอาจหมดระหว่างเดือน (EOM=0)", "issueKey": issue_key})
            monthly_data_for_product.append({"monthYear": row['month_year'], "sold": convert_value(current_sales), "revenue": convert_value(row['total_revenue']), "bomStock": convert_value(bom_stock), "eomStock": convert_value(eom_stock), "analysis": analysis_text})
        current_product_overall = overall_sales_by_product[overall_sales_by_product['item_code'] == item_code_val]
        products_data_list.append({"itemCode": item_code_val, "itemName": item_name_val, "totalSold": convert_value(current_product_overall['overall_quantity'].sum()), "totalRevenue": convert_value(current_product_overall['overall_revenue'].sum()), "monthlyData": monthly_data_for_product, "isBestSeller": item_code_val in best_sellers_overall_df['item_code'].values, "isSlowMover": item_code_val in slow_movers_overall_df['item_code'].values})
    for issue in potential_stock_issues: issue.pop('issueKey', None)

    # --- Branch Analysis (Modified for Monthly Filtering) ---
    branch_analysis_output = {
        "monthlyOverallSalesByBranch": {}, # Key: monthYear or "all", Value: list of branch sales
        "monthlyTopProductsPerBranch": {}, # Key: monthYear or "all", Value: list of top products per branch
        "trendData": [] # Original trend data for line chart (remains unchanged in structure)
    }

    if 'branch' in sales_df.columns and not sales_df.empty:
        # Overall (All Months) Branch Sales
        overall_sales_by_branch_df = sales_df.groupby('branch').agg(total_revenue=('revenue', 'sum'), total_quantity=('quantity', 'sum')).reset_index().sort_values(by='total_revenue', ascending=False)
        branch_analysis_output["monthlyOverallSalesByBranch"]["all"] = [{"branch": row['branch'], "total_revenue": convert_value(row['total_revenue']), "total_quantity": convert_value(row['total_quantity'])} for _, row in overall_sales_by_branch_df.iterrows()]

        # Overall (All Months) Top Products Per Branch
        branch_item_sales_overall = sales_df.groupby(['branch', 'item_code', 'item_name']).agg(revenue_sum=('revenue', 'sum'), quantity_sum=('quantity', 'sum')).reset_index()
        top_products_overall_list = []
        for branch_name, group in branch_item_sales_overall.groupby('branch'):
            top_prods_df = group.sort_values(by='revenue_sum', ascending=False).head(5)
            top_products_overall_list.append({"branch": branch_name, "topProducts": [{"item_code": r['item_code'], "item_name": r['item_name'], "revenue": convert_value(r['revenue_sum']), "quantity": convert_value(r['quantity_sum'])} for _, r in top_prods_df.iterrows()]})
        branch_analysis_output["monthlyTopProductsPerBranch"]["all"] = top_products_overall_list
        
        # Monthly Branch Sales and Top Products
        for month in active_months:
            month_sales_branch_df = sales_df[sales_df['month_year'] == month]
            if not month_sales_branch_df.empty:
                # Monthly Overall Sales by Branch
                overall_sales_month_branch_df = month_sales_branch_df.groupby('branch').agg(total_revenue=('revenue', 'sum'), total_quantity=('quantity', 'sum')).reset_index().sort_values(by='total_revenue', ascending=False)
                branch_analysis_output["monthlyOverallSalesByBranch"][month] = [{"branch": row['branch'], "total_revenue": convert_value(row['total_revenue']), "total_quantity": convert_value(row['total_quantity'])} for _, row in overall_sales_month_branch_df.iterrows()]
                
                # Monthly Top Products Per Branch
                branch_item_sales_month = month_sales_branch_df.groupby(['branch', 'item_code', 'item_name']).agg(revenue_sum=('revenue', 'sum'), quantity_sum=('quantity', 'sum')).reset_index()
                top_products_month_list = []
                for branch_name, group in branch_item_sales_month.groupby('branch'):
                    top_prods_df = group.sort_values(by='revenue_sum', ascending=False).head(5)
                    top_products_month_list.append({"branch": branch_name, "topProducts": [{"item_code": r['item_code'], "item_name": r['item_name'], "revenue": convert_value(r['revenue_sum']), "quantity": convert_value(r['quantity_sum'])} for _, r in top_prods_df.iterrows()]})
                branch_analysis_output["monthlyTopProductsPerBranch"][month] = top_products_month_list
            else: # Ensure keys exist even if no data for that month
                branch_analysis_output["monthlyOverallSalesByBranch"][month] = []
                branch_analysis_output["monthlyTopProductsPerBranch"][month] = []


        # Trend Data (remains the same structure as before, for the line chart)
        monthly_sales_by_branch_agg = sales_df.groupby(['month_year', 'branch']).agg(revenue=('revenue', 'sum'), quantity=('quantity', 'sum')).reset_index().sort_values(['branch', 'month_year'])
        trend_data_list = []
        for branch_name, group in monthly_sales_by_branch_agg.groupby('branch'):
            monthly_data = [{"monthYear": r['month_year'], "revenue": convert_value(r['revenue']), "quantity": convert_value(r['quantity'])} for _, r in group.iterrows()]
            trend_data_list.append({"branch": branch_name, "monthlyData": monthly_data})
        branch_analysis_output["trendData"] = trend_data_list
    else: # Default empty structures if no branch data
        branch_analysis_output = {"monthlyOverallSalesByBranch": {"all": []}, "monthlyTopProductsPerBranch": {"all": []}, "trendData": []}
        for month in active_months:
            branch_analysis_output["monthlyOverallSalesByBranch"][month] = []
            branch_analysis_output["monthlyTopProductsPerBranch"][month] = []


    monthly_summary_analysis_list = [] # (Monthly summary logic - assuming correct)
    if not master_df.empty and active_months:
        for month in active_months:
            month_sales_df, month_master_df = sales_df[sales_df['month_year'] == month], master_df[master_df['month_year'] == month]
            total_revenue_month, total_quantity_month, unique_products_sold_month = month_sales_df['revenue'].sum(), month_sales_df['quantity'].sum(), month_sales_df['item_code'].nunique()
            top_selling_month_rev = month_sales_df.groupby(['item_code', 'item_name'])['revenue'].sum().nlargest(5).reset_index()
            top_selling_month_qty = month_sales_df.groupby(['item_code', 'item_name'])['quantity'].sum().nlargest(5).reset_index()
            products_bom_stock_out = month_master_df[month_master_df['bom_stock'] == 0]['item_code'].nunique()
            products_eom_stock_out_with_sales = month_master_df[(month_master_df['eom_stock'] == 0) & ((month_master_df['total_quantity_sold'] > 0) | (month_master_df['bom_stock'] > 0))]['item_code'].nunique()
            monthly_summary_analysis_list.append({"monthYear": month, "totalRevenue": convert_value(total_revenue_month), "totalQuantity": convert_value(total_quantity_month), "uniqueProductsSold": convert_value(unique_products_sold_month), "topSellingByRevenue": [{"itemCode": r['item_code'], "itemName": r['item_name'], "revenue": convert_value(r['revenue'])} for _, r in top_selling_month_rev.iterrows()], "topSellingByQuantity": [{"itemCode": r['item_code'], "itemName": r['item_name'], "quantity": convert_value(r['quantity'])} for _, r in top_selling_month_qty.iterrows()], "productsWithBomStockOut": convert_value(products_bom_stock_out), "productsWithEomStockOut": convert_value(products_eom_stock_out_with_sales)})

    mom_comparison_list = [] # (MoM comparison logic - assuming correct)
    if len(active_months) > 1:
        overall_monthly_sales = sales_df.groupby('month_year').agg(total_revenue=('revenue', 'sum'), total_quantity=('quantity', 'sum')).reset_index().sort_values(by='month_year')
        overall_monthly_sales['prev_revenue'] = overall_monthly_sales['total_revenue'].shift(1); overall_monthly_sales['prev_quantity'] = overall_monthly_sales['total_quantity'].shift(1)
        for index, row in overall_monthly_sales.iterrows():
            if index == 0: continue
            current_m, prev_m_data = row['month_year'], overall_monthly_sales.iloc[index-1]
            revenue_change_abs, quantity_change_abs = row['total_revenue'] - prev_m_data['total_revenue'], row['total_quantity'] - prev_m_data['total_quantity']
            revenue_change_pct = (revenue_change_abs / prev_m_data['total_revenue']) * 100 if prev_m_data['total_revenue'] != 0 else 0
            quantity_change_pct = (quantity_change_abs / prev_m_data['total_quantity']) * 100 if prev_m_data['total_quantity'] != 0 else 0
            mom_notes = [f"ยอดขายรวม {'เพิ่มขึ้น' if revenue_change_pct > 0 else 'ลดลง' if revenue_change_pct < 0 else 'ไม่เปลี่ยนแปลง'} {abs(revenue_change_pct):.2f}%", f"จำนวนขายรวม {'เพิ่มขึ้น' if quantity_change_pct > 0 else 'ลดลง' if quantity_change_pct < 0 else 'ไม่เปลี่ยนแปลง'} {abs(quantity_change_pct):.2f}%"]
            current_month_stock_issues = [f"สินค้าขาดสต็อก: {psi['itemName']} ({psi['issue']})" for psi in potential_stock_issues if psi['monthYear'] == current_m]
            if current_month_stock_issues: mom_notes.extend(["ประเด็นสต็อกในเดือนปัจจุบัน:"] + current_month_stock_issues[:3])
            mom_comparison_list.append({"currentMonth": current_m, "previousMonth": prev_m_data['month_year'], "overallRevenueCurrent": convert_value(row['total_revenue']), "overallRevenuePrevious": convert_value(prev_m_data['total_revenue']), "overallRevenueChangeAbsolute": convert_value(revenue_change_abs), "overallRevenueChangePercentage": convert_value(revenue_change_pct), "overallQuantityCurrent": convert_value(row['total_quantity']), "overallQuantityPrevious": convert_value(prev_m_data['total_quantity']), "overallQuantityChangeAbsolute": convert_value(quantity_change_abs), "overallQuantityChangePercentage": convert_value(quantity_change_pct), "notes": mom_notes})

    monthly_analysis_insights_dict = {} # (Monthly insights logic - assuming correct)
    if not master_df.empty and active_months:
        for month in active_months:
            month_master_df = master_df[master_df['month_year'] == month]
            month_sales_agg = month_master_df.groupby(['item_code', 'item_name']).agg(metric_revenue=('total_revenue', 'sum'), metric_quantity=('total_quantity_sold', 'sum')).reset_index()
            best_sellers_month_df = month_sales_agg.sort_values(by='metric_revenue', ascending=False).head(10)
            month_master_df_for_slow = month_master_df.copy(); month_master_df_for_slow['metric_avg_sales'] = month_master_df_for_slow['total_quantity_sold']; month_master_df_for_slow['metric_avg_stock'] = month_master_df_for_slow['bom_stock']
            slow_movers_month_df = month_master_df_for_slow[(month_master_df_for_slow['metric_avg_sales'] < 10) & (month_master_df_for_slow['metric_avg_stock'] > 0)].sort_values(by='metric_avg_sales', ascending=True).head(10)
            monthly_analysis_insights_dict[month] = {"bestSellers": [{"item_code": r['item_code'], "item_name": r['item_name'], "metric_revenue": convert_value(r['metric_revenue']), "metric_quantity": convert_value(r['metric_quantity'])} for _, r in best_sellers_month_df.iterrows()], "slowMovers": [{"item_code": r['item_code'], "item_name": r['item_name'], "metric_avg_sales": convert_value(r['metric_avg_sales']), "metric_avg_stock": convert_value(r['metric_avg_stock'])} for _, r in slow_movers_month_df.iterrows()]}

    key_indicators = [] # (Key indicators logic - assuming correct)
    total_revenue_all, total_quantity_all = sales_df['revenue'].sum(), sales_df['quantity'].sum()
    monthly_sales_trend_records = sales_df.groupby('month_year').agg(revenue=('revenue', 'sum'), quantity=('quantity', 'sum')).reset_index().sort_values('month_year')
    monthly_sales_trend_for_chart_cleaned = [{"monthYear": row['month_year'], "revenue": convert_value(row['revenue']), "quantity": convert_value(row['quantity'])} for _, row in monthly_sales_trend_records.iterrows()]
    if not stock_df.empty and total_quantity_all > 0 and not master_df['bom_stock'].dropna().empty:
        avg_total_bom_stock_per_month = master_df.groupby('month_year')['bom_stock'].sum().mean()
        if avg_total_bom_stock_per_month > 0: key_indicators.append({"name": "อัตราส่วนหมุนเวียนสินค้าคงคลัง (ภาพรวม)", "value": f"{convert_value(total_quantity_all / avg_total_bom_stock_per_month):.2f} รอบ", "details": "ยอดขายรวม (ชิ้น) / สต็อกเฉลี่ยต้นเดือนรวม"})
        else: key_indicators.append({"name": "อัตราส่วนหมุนเวียนสินค้าคงคลัง (ภาพรวม)", "value": "N/A", "details": "สต็อกเฉลี่ยเป็นศูนย์"})
    else: key_indicators.append({"name": "อัตราส่วนหมุนเวียนสินค้าคงคลัง (ภาพรวม)", "value": "N/A", "details": "ไม่มีข้อมูลสต็อก"})
    if not stock_df.empty and total_quantity_all > 0 and active_months and not master_df.empty:
        first_month = active_months[0]; initial_stock_sum_for_sell_through = master_df[master_df['month_year'] == first_month]['bom_stock'].sum()
        denominator_sell_through = total_quantity_all + initial_stock_sum_for_sell_through
        if denominator_sell_through > 0 : key_indicators.append({"name": "อัตราการขายสินค้าผ่าน (Sell-Through Rate - ภาพรวม)", "value": f"{convert_value((total_quantity_all / denominator_sell_through) * 100):.2f}%", "details": "(ยอดขายรวม (ชิ้น) / (ยอดขายรวม (ชิ้น) + สต็อกรวมต้นงวดเดือนแรก)) * 100"})
        else: key_indicators.append({"name": "อัตราการขายสินค้าผ่าน (Sell-Through Rate - ภาพรวม)", "value": "N/A", "details": "ผลรวมยอดขายและสต็อกเริ่มต้นเป็นศูนย์"})
    else: key_indicators.append({"name": "อัตราการขายสินค้าผ่าน (Sell-Through Rate - ภาพรวม)", "value": "N/A", "details": "ไม่มีข้อมูลสต็อกหรือยอดขาย"})

    output_json = {
        "summary": {"totalRevenue": convert_value(total_revenue_all), "totalQuantitySold": convert_value(total_quantity_all), "activeMonths": active_months, "dataSourceNote": "ข้อมูลประมวลผล (เพิ่ม Filter เดือนใน Branch Analysis)"},
        "monthlySalesTrend": monthly_sales_trend_for_chart_cleaned,
        "productsData": products_data_list,
        "analysisInsights": {"bestSellers": [{"item_code": r['item_code'], "item_name": r['item_name'], "metric_revenue": convert_value(r['overall_revenue']), "metric_quantity": convert_value(r['overall_quantity'])} for _, r in best_sellers_overall_df.iterrows()], "slowMovers": [{"item_code": r['item_code'], "item_name": r['item_name'], "metric_avg_sales": convert_value(r['avg_monthly_sales']), "metric_avg_stock": convert_value(r['avg_bom_stock'])} for _, r in slow_movers_overall_df.iterrows()], "potentialStockIssues": potential_stock_issues },
        "monthlyAnalysisInsights": monthly_analysis_insights_dict, 
        "branchAnalysis": branch_analysis_output, # Updated structure
        "monthlySummaryAnalysis": monthly_summary_analysis_list, 
        "monthOverMonthComparison": mom_comparison_list, 
        "keyIndicators": key_indicators
    }
    return output_json

if __name__ == '__main__':
    dashboard_data = process_sales_data()
    if dashboard_data:
        try:
            with open('dashboard.json', 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            print("dashboard.json has been generated successfully.")
        except TypeError as e: print(f"JSON serialization error: {e}")
    else: print("Failed to generate dashboard data.")

