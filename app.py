import warnings
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from scipy import stats
import os

BASE_URL = "https://banks.data.fdic.gov/api"

class BankDataAnalyzer:
    def __init__(self):
        self.institutions_data = {}
        self.financials_data = {}
        self.dollar_format_metrics = [
            'Total Assets', 'Total Deposits', 'Total Loans and Leases', 'Net Loans and Leases',
            'Total Securities', 'Real Estate Loans', 'Loans to Residential Properties', 'Multifamily',
            'Farmland Real Estate Loans', 'Loans to Nonresidential Properties',
            'Owner-Occupied Nonresidential Properties Loans', 'Non-OOC Nonresidential Properties Loans',
            'RE Construction and Land Development', '1-4 Family Residential Construction and Land Development Loans',
            'Other Construction, All Land Development and Other Land Loans',
            'Commercial Real Estate Loans not Secured by Real Estate', 'Commercial and Industrial Loans',
            'Agriculture Loans', 'Credit Cards', 'Consumer Loans', 'Allowance for Credit Loss',
            'Past Due 30-89 Days', 'Past Due 90+ Days', 'Tier 1 (Core) Capital',
            'Total Charge-Offs', 'Total Recoveries', 'Net Income',
            'Total Loans and Leases Net Charge-Offs Quarterly', 'Common Equity Tier 1 Before Adjustments',
            'Bank Equity Capital', 'CECL Transition Amount', 'Perpetual Preferred Stock'
        ]
        self.metric_definitions = {
            'Total Assets': "(YTD, $) The sum of all assets owned by the entity.",
            'Total Deposits': "(YTD, $) The sum of all deposits including demand, savings, and time deposits.",
            'Total Loans and Leases': "(YTD, $) Total loans and lease financing receivables.",
            'Net Loans and Leases': "(YTD, $) Net Loans and Leases",
            'Total Securities': "(YTD, $) Sum of held-to-maturity, available-for-sale, and equity securities.",
            'Real Estate Loans': "(YTD, $) Loans primarily secured by real estate.",
            'Loans to Residential Properties': "(YTD, $) Total loans for residential properties.",
            'Multifamily': "(YTD, $) Loans for multifamily residential properties.",
            'Farmland Real Estate Loans': "(YTD, $) Loans secured by farmland.",
            '1-4 Family Residential Construction and Land Development Loans': "(YTD, $) Construction and land development loans for 1-4 family residential properties.",
            'Other Construction, All Land Development and Other Land Loans': "(YTD, $) Other construction loans, all land development and other land loans.",
            'Loans to Nonresidential Properties': "(YTD, $) Total loans for nonresidential properties.",
            'Owner-Occupied Nonresidential Properties Loans': "(YTD, $) Loans for owner-occupied nonresidential properties.",
            'Non-OOC Nonresidential Properties Loans': "(YTD, $) Loans for non-owner-occupied nonresidential properties.",
            'Commercial Real Estate Loans not Secured by Real Estate': "(YTD, $) Commercial real estate loans that are not secured by real estate.",
            'Commercial and Industrial Loans': "(YTD, $) Loans for commercial and industrial purposes, excluding real estate-secured loans.",
            'Agriculture Loans': "(YTD, $) Loans to finance agricultural production and other loans to farmers.",
            'Credit Cards': "(YTD, $) Consumer loans extended through credit card plans.",
            'Consumer Loans': "(YTD, $) Other loans to individuals for personal expenditures, including student loans.",
            'Allowance for Credit Loss': "(YTD, $) Reserve for estimated credit losses associated with the loan and lease portfolio.",
            'Past Due 30-89 Days': "(Qtly, $) Loans and leases past due 30-89 days, in dollars.",
            'Past Due 90+ Days': "(Qtly, $) Loans and leases past due 90 days or more, in dollars.",
            'Tier 1 (Core) Capital': "(Qtly, $) Tier 1 core capital, which includes common equity tier 1 capital and additional tier 1 capital.",
            'Total Charge-Offs': "(YTD, $) Total charge-offs of loans and leases.",
            'Total Recoveries': "(YTD, $) Total recoveries of loans and leases previously charged off.",
            'Total Loans and Leases Net Charge-Offs Quarterly': "(Qtly, $) Total loans and leases net charge-offs for the quarter.",
            'Net Income': "(YTD, $) Net income earned by the entity.",
            'RE Construction and Land Development': "(YTD, $) Real estate construction and land development loans.",
            'RE Construction and Land Development to Tier 1 + ACL': "(Qtly, %) Real estate construction and land development loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Common Equity Tier 1 Before Adjustments': "(YTD, $) Common Equity Tier 1 capital before adjustments.",
            'Bank Equity Capital': "(YTD, $) Total bank equity capital.",
            'Perpetual Preferred Stock': "(YTD, $) The amount of perpetual preferred stock issued by the bank.",
            'CECL Transition Amount': "(YTD, $) Current Expected Credit Loss (CECL) Transition Amount, not including Deferred Tax Assets, adjusted for Perpetual Preferred Stock.",
            'Net Interest Margin': "(YTD, %) The net interest margin of the entity.",
            'Earning Assets / Total Assets': "(Qtly, %) Ratio of earning assets to total assets.",
            'Nonperforming Assets / Total Assets': "(Qtly, %) Ratio of nonperforming assets to total assets.",
            'Assets Past Due 30-89 Days / Total Assets': "(Qtly, %) Ratio of assets past due 30-89 days to total assets.",
            'Assets Past Due 90+ Days / Total Assets': "(Qtly, %) Ratio of assets past due 90+ days to total assets.",
            'Net Charge-Offs / Total Loans & Leases': "(YTD, %) Ratio of net charge-offs to total loans and leases.",
            'Earnings Coverage of Net Loan Charge-Offs': "(X) The number of times that earnings can cover net loan charge-offs.",
            'Loan and Lease Loss Provision to Net Charge-Offs': "(YTD, %) Ratio of loan loss provision to net charge-offs.",
            'Loss Allowance / Total Loans & Leases': "(YTD, %) Ratio of loss allowance to total loans and leases.",
            'Loss Allowance to Noncurrent Loans and Leases': "(Qtly, %) Ratio of loss allowance to noncurrent loans and leases.",
            'Noncurrent Loans / Total Loans': "(Qtly, %) Ratio of noncurrent loans to total loans.",
            'Net Loans and Leases to Deposits': "(YTD, %) Loans and lease financing receivables net of unearned income, allowances and reserves as a percent of total deposits.",
            'Net Loans and Leases to Assets': "(Qtly, %) Ratio of net loans and leases to assets.",
            'Return on Assets': "(YTD, %) Return on assets.",
            'Return on Equity': "(YTD, %) Return on equity.",
            'Leverage (Core Capital) Ratio': "(Qtly, %) Leverage ratio (core capital ratio).",
            'Total Risk-Based Capital Ratio': "(Qtly, %) Total risk-based capital ratio.",
            'Efficiency Ratio': "(YTD, %) The efficiency ratio of the entity.",
            'Real Estate Loans to Tier 1 + ACL': "(Qtly, %) Real Estate Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Commercial RE to Tier 1 + ACL': "(Qtly, %) Sum of RE Construction and Land Development, Multifamily, Loans to Nonresidential Properties, and Commercial Real Estate Loans not Secured by Real Estate as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Non-Owner Occupied CRE 3-Year Growth Rate': "(%) 3-year annualized growth rate of Non-Owner Occupied Commercial Real Estate...",
            'C&I Loans to Tier 1 + ACL': "(Qtly, %) Commercial and Industrial Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Agriculture Loans to Tier 1 + ACL': "(Qtly, %) Agriculture Loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Credit Cards to Tier 1 + ACL': "(Qtly, %) Credit Card loans as a percentage of Tier 1 (Core) Capital plus Allowance for Credit Loss.",
            'Net Charge-Offs / Allowance for Credit Loss': "(Qtly, %) Ratio of Quarterly Net Charge-Offs to Allowance for Credit Loss."
        }

    def get_data(self, endpoint: str, params: Dict) -> Dict:
        url = f"{BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, params=params, headers={"Accept": "application/json"}, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data: {e}")
            return {"data": []}

    def get_institutions(self, filters: str = "", fields: str = "") -> List[Dict]:
        params = {"filters": filters, "fields": fields, "limit": 10000}
        data = self.get_data("institutions", params)
        return data.get('data', [])

    def get_financials(self, cert: str, filters: str = "", fields: str = "") -> List[Dict]:
        params = {"filters": f"CERT:{cert}" + (f" AND {filters}" if filters else ""), "fields": fields, "limit": 10000}
        data = self.get_data("financials", params)
        return data.get('data', [])

    def fetch_data(self, bank_info: List[Union[str, Dict]], start_date: str, end_date: str):
        institution_fields = "NAME,CERT"
        financial_fields = (
            "CERT,REPDTE,ASSET,DEP,LNLSGR,LNLSNET,SC,LNRE,LNCI,LNAG,LNCRCD,LNCONOTH,LNATRES,"
            "P3ASSET,P9ASSET,RBCT1J,DRLNLS,CRLNLS,NETINC,ERNASTR,NPERFV,P3ASSETR,P9ASSETR,"
            "NIMY,NTLNLSR,LNATRESR,NCLNLSR,ROA,ROE,RBC1AAJ,RBCT2,RBCRWAJ,LNLSDEPR,"
            "LNLSNTV,EEFFR,LNRESNCR,ELNANTR,IDERNCVR,NTLNLSQ,LNRECONS,"
            "LNRENRES,LNRENROW,LNRENROT,LNRERES,LNREMULT,LNREAG,LNRECNFM,LNRECNOT,LNCOMRE,CT1BADJ,EQ,EQPP"
        )

        for bank_item in bank_info:
            if isinstance(bank_item, str):
                institutions = self.get_institutions(f'NAME:"{bank_item}"', institution_fields)
            elif isinstance(bank_item, dict):
                institutions = self.get_institutions(f'CERT:{bank_item["cert"]}', institution_fields)
            else:
                print(f"Invalid bank info format: {bank_item}")
                continue

            if institutions:
                bank = institutions[0]
                if isinstance(bank, dict) and 'data' in bank:
                    bank_data = bank['data']
                    if 'NAME' in bank_data and 'CERT' in bank_data:
                        self.institutions_data[bank_data['NAME']] = bank_data
                        financials = self.get_financials(
                            bank_data['CERT'], 
                            f"REPDTE:[{start_date} TO {end_date}]", 
                            fields=financial_fields
                        )
                        self.financials_data[bank_data['NAME']] = [
                            f['data'] for f in financials if isinstance(f, dict) and 'data' in f
                        ]
                    else:
                        print(f"Warning: Required fields missing for bank: {bank_item}")
                else:
                    print(f"Unexpected data structure for bank: {bank_item}")
            else:
                print(f"No data found for bank: {bank_item}")

    def safe_float(self, value):
        try:
            return float(value) if value is not None else 0.0
        except ValueError:
            return 0.0

    def calculate_metrics(self) -> pd.DataFrame:
        all_metrics = []

        for bank_name, financials in self.financials_data.items():
            sorted_financials = sorted(financials, key=lambda x: x['REPDTE'])

            for i, financial in enumerate(sorted_financials):
                metrics = {
                    'Bank': bank_name,
                    'Date': financial.get('REPDTE'),
                    'Total Assets': self.safe_float(financial.get('ASSET')),
                    'Total Deposits': self.safe_float(financial.get('DEP')),
                    'Total Loans and Leases': self.safe_float(financial.get('LNLSGR')),
                    'Net Loans and Leases': self.safe_float(financial.get('LNLSNET')),
                    'Total Securities': self.safe_float(financial.get('SC')),
                    'Real Estate Loans': self.safe_float(financial.get('LNRE')),
                    'Loans to Residential Properties': self.safe_float(financial.get('LNRERES')),
                    'Multifamily': self.safe_float(financial.get('LNREMULT')),
                    'Farmland Real Estate Loans': self.safe_float(financial.get('LNREAG')),
                    'Loans to Nonresidential Properties': self.safe_float(financial.get('LNRENRES')),
                    'Owner-Occupied Nonresidential Properties Loans': self.safe_float(financial.get('LNRENROW')),
                    'Non-OOC Nonresidential Properties Loans': self.safe_float(financial.get('LNRENROT')),
                    'RE Construction and Land Development': self.safe_float(financial.get('LNRECONS')),
                    '1-4 Family Residential Construction and Land Development Loans': self.safe_float(financial.get('LNRECNFM')),
                    'Other Construction, All Land Development and Other Land Loans': self.safe_float(financial.get('LNRECNOT')),
                    'Commercial Real Estate Loans not Secured by Real Estate': self.safe_float(financial.get('LNCOMRE')),
                    'Commercial and Industrial Loans': self.safe_float(financial.get('LNCI')),
                    'Agriculture Loans': self.safe_float(financial.get('LNAG')),
                    'Credit Cards': self.safe_float(financial.get('LNCRCD')),
                    'Consumer Loans': self.safe_float(financial.get('LNCONOTH')),
                    'Allowance for Credit Loss': self.safe_float(financial.get('LNATRES')),
                    'Past Due 30-89 Days': self.safe_float(financial.get('P3ASSET')),
                    'Past Due 90+ Days': self.safe_float(financial.get('P9ASSET')),
                    'Tier 1 (Core) Capital': self.safe_float(financial.get('RBCT1J')),
                    'Total Charge-Offs': self.safe_float(financial.get('DRLNLS')),
                    'Total Recoveries': self.safe_float(financial.get('CRLNLS')),
                    'Total Loans and Leases Net Charge-Offs Quarterly': self.safe_float(financial.get('NTLNLSQ')),
                    'Net Income': self.safe_float(financial.get('NETINC')),
                    'Common Equity Tier 1 Before Adjustments': self.safe_float(financial.get('CT1BADJ')),
                    'Bank Equity Capital': self.safe_float(financial.get('EQ')),
                    'Perpetual Preferred Stock': self.safe_float(financial.get('EQPP')),
                    'Net Interest Margin': self.safe_float(financial.get('NIMY')),
                    'Earning Assets / Total Assets': self.safe_float(financial.get('ERNASTR')),
                    'Nonperforming Assets / Total Assets': self.safe_float(financial.get('NPERFV')),
                    'Assets Past Due 30-89 Days / Total Assets': self.safe_float(financial.get('P3ASSETR')),
                    'Assets Past Due 90+ Days / Total Assets': self.safe_float(financial.get('P9ASSETR')),
                    'Net Charge-Offs / Total Loans & Leases': self.safe_float(financial.get('NTLNLSR')),
                    'Earnings Coverage of Net Loan Charge-Offs': self.safe_float(financial.get('IDERNCVR')),
                    'Loan and Lease Loss Provision to Net Charge-Offs': self.safe_float(financial.get('ELNANTR')),
                    'Loss Allowance / Total Loans & Leases': self.safe_float(financial.get('LNATRESR')),
                    'Loss Allowance to Noncurrent Loans and Leases': self.safe_float(financial.get('LNRESNCR')),
                    'Noncurrent Loans / Total Loans': self.safe_float(financial.get('NCLNLSR')),
                    'Net Loans and Leases to Deposits': self.safe_float(financial.get('LNLSDEPR')),
                    'Net Loans and Leases to Assets': self.safe_float(financial.get('LNLSNTV')),
                    'Return on Assets': self.safe_float(financial.get('ROA')),
                    'Return on Equity': self.safe_float(financial.get('ROE')),
                    'Leverage (Core Capital) Ratio': self.safe_float(financial.get('RBC1AAJ')),
                    'Total Risk-Based Capital Ratio': self.safe_float(financial.get('RBCRWAJ')),
                    'Efficiency Ratio': self.safe_float(financial.get('EEFFR'))
                }

                ct1badj = metrics['Common Equity Tier 1 Before Adjustments']
                eq = metrics['Bank Equity Capital']
                eqpp = metrics['Perpetual Preferred Stock']
                tier1_capital = metrics['Tier 1 (Core) Capital']
                allowance_for_credit_loss = metrics['Allowance for Credit Loss']

                date = pd.to_datetime(financial.get('REPDTE'), format='%Y%m%d')
                if date >= pd.to_datetime('2019-01-01'):
                    cecl_transition_amount = ct1badj - eq + eqpp
                    metrics['CECL Transition Amount'] = cecl_transition_amount
                    capital_base = tier1_capital + allowance_for_credit_loss - cecl_transition_amount
                else:
                    metrics['CECL Transition Amount'] = 0
                    capital_base = tier1_capital + allowance_for_credit_loss

                if capital_base > 0:
                    metrics['Real Estate Loans to Tier 1 + ACL'] = (metrics['Real Estate Loans'] / capital_base) * 100
                    metrics['RE Construction and Land Development to Tier 1 + ACL'] = (metrics['RE Construction and Land Development'] / capital_base) * 100
                    metrics['C&I Loans to Tier 1 + ACL'] = (metrics['Commercial and Industrial Loans'] / capital_base) * 100
                    metrics['Agriculture Loans to Tier 1 + ACL'] = (metrics['Agriculture Loans'] / capital_base) * 100
                    metrics['Credit Cards to Tier 1 + ACL'] = (metrics['Credit Cards'] / capital_base) * 100

                    commercial_re = (
                        metrics['RE Construction and Land Development'] +
                        metrics['Multifamily'] +
                        metrics['Loans to Nonresidential Properties'] +
                        metrics['Commercial Real Estate Loans not Secured by Real Estate']
                    )
                    metrics['Commercial RE to Tier 1 + ACL'] = (commercial_re / capital_base) * 100
                else:
                    metrics['Real Estate Loans to Tier 1 + ACL'] = 0
                    metrics['RE Construction and Land Development to Tier 1 + ACL'] = 0
                    metrics['C&I Loans to Tier 1 + ACL'] = 0
                    metrics['Agriculture Loans to Tier 1 + ACL'] = 0
                    metrics['Commercial RE to Tier 1 + ACL'] = 0
                    metrics['Credit Cards to Tier 1 + ACL'] = 0

                non_owner_occupied_cre = (
                    self.safe_float(financial.get('LNRECONS')) +
                    self.safe_float(financial.get('LNREMULT')) +
                    self.safe_float(financial.get('LNRENROT')) +
                    self.safe_float(financial.get('LNCOMRE'))
                )

                if i >= 12:
                    three_years_ago = sorted_financials[i-12]
                    old_non_owner_occupied_cre = (
                        self.safe_float(three_years_ago.get('LNRECONS')) +
                        self.safe_float(three_years_ago.get('LNREMULT')) +
                        self.safe_float(three_years_ago.get('LNRENROT')) +
                        self.safe_float(three_years_ago.get('LNCOMRE'))
                    )
                    if old_non_owner_occupied_cre > 0:
                        growth_rate = (non_owner_occupied_cre / old_non_owner_occupied_cre) - 1
                        metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = growth_rate * 100
                    else:
                        metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None
                else:
                    metrics['Non-Owner Occupied CRE 3-Year Growth Rate'] = None

                if metrics['Allowance for Credit Loss'] > 0:
                    metrics['Net Charge-Offs / Allowance for Credit Loss'] = (metrics['Total Loans and Leases Net Charge-Offs Quarterly'] / metrics['Allowance for Credit Loss']) * 100
                else:
                    metrics['Net Charge-Offs / Allowance for Credit Loss'] = 0

                all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        return df.sort_values('Date')

def create_dashboard(df, dollar_format_metrics, metric_definitions, start_date, end_date):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    server = app.server  

    start_datetime = pd.to_datetime(start_date, format='%Y%m%d')
    end_datetime = pd.to_datetime(end_date, format='%Y%m%d')

    df['Date'] = pd.to_datetime(df['Date']).dt.to_pydatetime()
    unique_dates = sorted(df['Date'].unique())
    unique_dates = [pd.to_datetime(date).to_pydatetime() for date in unique_dates]

    metric_order = [
        'Real Estate Loans to Tier 1 + ACL',
        'RE Construction and Land Development to Tier 1 + ACL',
        'Commercial RE to Tier 1 + ACL',
        'Non-Owner Occupied CRE 3-Year Growth Rate',
        'C&I Loans to Tier 1 + ACL',
        'Agriculture Loans to Tier 1 + ACL',
        'Credit Cards to Tier 1 + ACL',
        'Net Charge-Offs / Allowance for Credit Loss',
        'Net Charge-Offs / Total Loans & Leases',
        'Earnings Coverage of Net Loan Charge-Offs',
        'Loan and Lease Loss Provision to Net Charge-Offs',
        'Loss Allowance / Total Loans & Leases',
        'Loss Allowance to Noncurrent Loans and Leases',
        'Nonperforming Assets / Total Assets',
        'Assets Past Due 30-89 Days / Total Assets',
        'Assets Past Due 90+ Days / Total Assets',
        'Noncurrent Loans / Total Loans',
        'Net Loans and Leases to Deposits',
        'Net Loans and Leases to Assets',
        'Return on Assets',
        'Return on Equity',
        'Leverage (Core Capital) Ratio',
        'Total Risk-Based Capital Ratio',
        'Efficiency Ratio',
        'Earning Assets / Total Assets',
        'Net Interest Margin'
    ] + dollar_format_metrics

    available_metrics = [metric for metric in metric_order if metric in df.columns]

    bank_name_mapping = {
        "Flagstar Bank, National Association": "Flagstar Bank",
        "The Huntington National Bank": "The Huntington National Bank",
        "Regions Bank": "Regions Bank",
        "Wells Fargo Bank, National Association": "Wells Fargo",
        "Bank of America, National Association": "Bank of America",
        "Citibank, National Association": "Citibank",
        "JPMorgan Chase Bank, National Association": "JPMorgan Chase",
        "U.S. Bank National Association": "U.S. Bank",
        "PNC Bank, National Association": "PNC Bank",
        "Truist Bank": "Truist Bank",
        "Goldman Sachs Bank USA": "Goldman Sachs",
        "Morgan Stanley Bank, National Association": "Morgan Stanley",
        "TD Bank, National Association": "TD Bank",
        "Capital One, National Association": "Capital One",
        "The Bank of New York Mellon": "BNY Mellon",
        "Fifth Third Bank, National Association": "Fifth Third Bank",
        "Citizens Bank, National Association": "Citizens Bank",
        "Ally Bank": "Ally Bank",
        "KeyBank National Association": "KeyBank",
        "Discover Bank": "Discover Bank",
        "Synchrony Bank": "Synchrony Bank",
        "Santander Bank, N.A.": "Santander Bank"
    }

    df['Bank'] = df['Bank'].map(bank_name_mapping)

    bank_peers = [
        "Bank of America",
        "Citibank",
        "JPMorgan Chase",
        "U.S. Bank",
        "PNC Bank",
        "Truist Bank",
        "Goldman Sachs",
        "Morgan Stanley",
        "TD Bank",
        "BNY Mellon",
        "Fifth Third Bank",
        "Citizens Bank",
        "KeyBank",
        "Santander Bank"
    ]

    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                /* CSS styles with goldish-yellowish theme for Flagstar Bank elements */
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #fdfaf0;
                    color: #333333;
                    margin: 0;
                    padding: 0;
                }
                #app-container {
                    display: flex;
                    height: 100vh;
                }
                .sidebar {
                    width: 500px;
                    background-color: #fff8e7;
                    padding: 2rem 1rem;
                    overflow-y: auto;
                    border-right: 1px solid #d1d1d1;
                }
                .content {
                    flex-grow: 1;
                    padding: 2rem;
                    overflow-y: auto;
                    background-color: #fdfaf0;
                }
                .card {
                    background-color: #fff;
                    border: none;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    margin-bottom: 1.5rem;
                }
                .card-title {
                    color: #333333;
                    margin-bottom: 0;
                    font-size: 1.1rem;
                    font-weight: bold;
                }
                .card-header {
                    background-color: #fff3d1;
                    border-bottom: 1px solid #d1d1d1;
                    padding: 0.5rem 1rem;
                }
                .card-body {
                    padding: 1rem;
                }
                .table {
                    font-size: 0.9rem;
                    background-color: #ffffff;
                }
                .table thead th {
                    background-color: #f7e7a9;
                    color: #333333;
                }
                .table tbody td {
                    color: #333333;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #fffaf0;
                }
                .table-hover tbody tr:hover {
                    background-color: #f5deb3;
                }
                .Select-menu-outer {
                    max-height: 400px !important;
                    background-color: #f5f5f5;
                }
                .Select-option {
                    padding: 12px 8px !important;
                    color: #333333;
                }
                .Select-option:hover {
                    background-color: #e0e0e0;
                }
                .Select-value-label {
                    color: #333333 !important;
                }
                .Select-control {
                    background-color: #f5f5f5 !important;
                    border-color: #d1d1d1 !important;
                }
                .Select-placeholder, .Select--single > .Select-control .Select-value {
                    color: #333333 !important;
                }
                .Select-input > input {
                    color: #333333 !important;
                }
                .source-info {
                    font-size: 0.8rem;
                    color: #666666;
                    text-align: center;
                    padding: 10px 0;
                }
                .metric-definition {
                    font-size: 0.9rem;
                    color: #666666;
                    margin-top: 10px;
                }
                .rc-slider-rail {
                    background-color: #bfbfbf;
                }
                .rc-slider-track {
                    background-color: #cd0000;
                }
                .rc-slider-handle {
                    border-color: #cd0000;
                }
                .rc-slider-mark-text {
                    color: #666666;
                }
                .stat-section {
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 5px;
                }
                .stat-section-title {
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #D4AF37;
                }
                .stat-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 5px;
                }
                .stat-label {
                    font-weight: bold;
                }
                .wf-highlight {
                    color: #D4AF37;
                }
                .add-all-btn {
                    background-color: #D4AF37;
                    color: #ffffff;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.9rem;
                    margin-top: 5px;
                    transition: background-color 0.3s;
                }
                .add-all-btn:hover {
                    background-color: #b29762;
                }
            </style>
            {%favicon%}
            {%css%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    sidebar = html.Div(
        [
            html.H4("Flagstar Bank Metrics Dashboard", className="display-6 mb-4", style={"color": "#333333"}),
            html.Hr(style={"borderColor": "#e9ecef"}),
            html.P("Select a metric to display", className="lead", style={"color": "#333333"}),
            dcc.Dropdown(
                id='metric-selector',
                options=[{'label': col, 'value': col, 'title': metric_definitions.get(col, '')} for col in available_metrics],
                value=available_metrics[0],
                clearable=False,
                style={'width': '100%', 'color': '#333333'},
                optionHeight=55
            ),
            html.Div(id='metric-definition', className="metric-definition mt-3"),
            html.Hr(style={"borderColor": "#e9ecef"}),
            html.P("Select peer groups to compare", className="lead", style={"color": "#333333"}),
            dcc.Checklist(
                id='peer-group-selector',
                options=[{'label': 'Bank Peers', 'value': 'bank_peers'}],
                value=['bank_peers'],
                style={'color': '#333333'}
            ),
            html.Div(id='peer-selector', className="mt-3"),
            html.Hr(style={"borderColor": "#e9ecef"}),
            html.P("Select trend timeline", className="lead", style={"color": "#333333"}),
            dcc.Dropdown(
                id='trend-timeline-selector',
                options=[
                    {'label': '1 Year', 'value': 1},
                    {'label': '2 Years', 'value': 2},
                    {'label': '5 Years', 'value': 5},
                    {'label': '10 Years', 'value': 10},
                    {'label': '15 Years', 'value': 15},
                    {'label': '20 Years', 'value': 20},
                ],
                value=5,
                clearable=False,
                style={'width': '100%', 'color': '#333333'},
            ),
            html.Div(id='selected-peers-info', className="mt-3", style={"color": "#333333"})
        ],
        className="sidebar"
    )

    content = html.Div([
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col(html.H4("Flagstar Bank vs Peer Banks", className="card-title"), width=8),
                    dbc.Col(
                        dcc.Dropdown(
                            id='date-selector',
                            options=[{'label': date.strftime('%m/%d/%y'), 'value': date.strftime('%Y-%m-%d')} 
                                     for date in unique_dates],
                            value=max(unique_dates).strftime('%Y-%m-%d'),
                            clearable=False,
                            style={'width': '120px', 'color': '#333333'},
                        ),
                        width=4,
                        style={'display': 'flex', 'justifyContent': 'flex-end'}
                    ),
                ])
            ]),
            dbc.CardBody([
                dcc.Graph(id='bar-chart')
            ])
        ]),
        dbc.Card([
            dbc.CardHeader(html.H4("Metric Overview", className="card-title")),
            dbc.CardBody([
                html.Div(id='metric-overview', className="p-0")
            ], className="p-0")
        ]),
        dbc.Card([
            dbc.CardHeader(html.H4("Bank Details", className="card-title")),
            dbc.CardBody([
                html.Div(id='bank-details')
            ])
        ]),
        html.Div("All data sourced through FDIC API", className="source-info")
    ], className="content")

    app.layout = html.Div([sidebar, content], id="app-container")

    @app.callback(
        Output('peer-selector', 'children'),
        Input('peer-group-selector', 'value')
    )
    def update_peer_selector(selected_groups):
        all_peers = []
        if 'bank_peers' in selected_groups:
            all_peers.extend(bank_peers)
        return dcc.Dropdown(
            id='individual-peer-selector',
            options=[{'label': peer, 'value': peer} for peer in all_peers],
            value=all_peers,
            multi=True,
            style={'width': '100%', 'color': '#333333'},
        )

    @app.callback(
        Output('bar-chart', 'figure'),
        Output('metric-overview', 'children'),
        Output('metric-definition', 'children'),
        Output('selected-peers-info', 'children'),
        Input('metric-selector', 'value'),
        Input('date-selector', 'value'),
        Input('individual-peer-selector', 'value'),
        Input('trend-timeline-selector', 'value')
    )
    def update_bar_chart(selected_metric, selected_date, selected_peers, trend_timeline):
        selected_date = pd.to_datetime(selected_date).to_pydatetime()
        selected_banks = ['Flagstar Bank'] + selected_peers
        filtered_df = df[(df['Date'] == selected_date) & (df['Bank'].isin(selected_banks))]

        if filtered_df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=f"No data available for {selected_date.strftime('%m/%d/%y')}",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="No data available for the selected date",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20)
                )]
            )
            empty_overview = html.Div("No data available for the selected date", style={"color": "#333333"})
            return empty_fig, empty_overview, "", f"No data available for {len(selected_banks)} selected banks"

        sorted_df = filtered_df.sort_values(by=selected_metric, ascending=False)
        colors = ['#D4AF37' if bank == 'Flagstar Bank' else '#808080' for bank in sorted_df['Bank']]

        fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}]], 
                            column_widths=[0.65, 0.35], horizontal_spacing=0.015)

        fig.add_trace(go.Bar(
            x=sorted_df['Bank'],
            y=sorted_df[selected_metric],
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>' + selected_metric + ': %{y:,.2f}<extra></extra>',
            showlegend=False
        ), row=1, col=1)

        trend_df = df[df['Bank'].isin(selected_banks)].pivot(index='Date', columns='Bank', values=selected_metric)
        start_date_offset = trend_df.index.max() - pd.DateOffset(years=trend_timeline)
        trend_df = trend_df[trend_df.index >= start_date_offset]

        for bank in trend_df.columns:
            color = '#D4AF37' if bank == 'Flagstar Bank' else '#808080'
            line_width = 3.5 if bank == 'Flagstar Bank' else 1.5
            opacity = 1 if bank == 'Flagstar Bank' else 0.4

            fig.add_trace(go.Scatter(
                x=trend_df.index,
                y=trend_df[bank],
                mode='lines',
                name=bank,
                line=dict(color=color, width=line_width),
                opacity=opacity,
                showlegend=False,
                hovertemplate='%{x|%m/%d/%y}<br>' + bank + ': %{y:,.2f}<extra></extra>'
            ), row=1, col=2)

        y_min = sorted_df[selected_metric].min()
        y_max = sorted_df[selected_metric].max()
        y_range = y_max - y_min
        y_padding = y_range * 0.1

        trend_min = trend_df.min().min()
        trend_max = trend_df.max().max()
        trend_range = trend_max - trend_min
        trend_padding = trend_range * 0.1

        formatted_date = selected_date.strftime('%m/%d/%y')

        fig.update_layout(
            xaxis2=dict(domain=[0.65, 1]),
            title=f"{selected_metric} as of {formatted_date}",
            title_x=0.01,
            height=600,
            margin=dict(l=50, r=20, t=50, b=100),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333333'),
            hoverlabel=dict(bgcolor="#ffffff", font_size=12, font_color="#333333"),
        )

        fig.update_xaxes(title_text=None, row=1, col=1, 
                         tickangle=-45, tickfont=dict(size=10),
                         showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', gridwidth=1)
        fig.update_yaxes(title_text=None, row=1, col=1,
                         tickformat=',.0f' if selected_metric in dollar_format_metrics else '.2f',
                         range=[y_min - y_padding, y_max + y_padding],
                         showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', gridwidth=1)

        if trend_timeline <= 2:
            dtick = 'M3'
            tickformat = '%b\n%Y'
        elif trend_timeline <= 5:
            dtick = 'M6'
            tickformat = '%b\n%Y'
        elif trend_timeline <= 10:
            dtick = 'M12'
            tickformat = '%Y'
        else:
            dtick = 'M24'
            tickformat = '%Y'

        fig.update_xaxes(title_text=None, row=1, col=2,
                         tickformat=tickformat, dtick=dtick,
                         showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', 
                         gridwidth=1, tickangle=-45)
        fig.update_yaxes(title_text=None, row=1, col=2,
                         range=[trend_min - trend_padding, trend_max + trend_padding],
                         showgrid=True, gridcolor='rgba(0, 0, 0, 0.1)', 
                         gridwidth=1,
                         tickformat=',.0f' if selected_metric in dollar_format_metrics else '.2f')

        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0.5, y=1.02,
            xanchor='center', yanchor='bottom',
            text=f"{trend_timeline}-Year Trend",
            showarrow=False,
            font=dict(size=12, color="#333333"),
            row=1, col=2
        )

        def format_value(value):
            if pd.isna(value):
                return "N/A"
            if selected_metric in dollar_format_metrics:
                return f"${value:,.0f}"
            else:
                return f"{value:.2f}"

        flagstar_value = filtered_df[filtered_df['Bank'] == 'Flagstar Bank'][selected_metric].values[0] if 'Flagstar Bank' in filtered_df['Bank'].values else None

        wf_percentile = stats.percentileofscore(filtered_df[selected_metric], flagstar_value) if flagstar_value is not None else None
        wf_rank = (filtered_df[selected_metric]
                   .rank(ascending=False, method='min')[filtered_df['Bank'] == 'Flagstar Bank']
                   .values[0] if flagstar_value is not None else None)
        q1, q3 = np.percentile(filtered_df[selected_metric], [25, 75])
        iqr = q3 - q1

        trend_df = df[df['Bank'].isin(selected_banks)].pivot(index='Date', columns='Bank', values=selected_metric)
        trend_df = trend_df[trend_df.index >= start_date_offset]
        flagstar_trend = trend_df['Flagstar Bank'].dropna() if 'Flagstar Bank' in trend_df.columns else pd.Series()
        peer_trends = trend_df.drop('Flagstar Bank', axis=1, errors='ignore').dropna()

        if not flagstar_trend.empty:
            wf_slope, _ = np.polyfit(range(len(flagstar_trend)), flagstar_trend, 1)
            wf_volatility = flagstar_trend.std()
        else:
            wf_slope = wf_volatility = np.nan

        if not peer_trends.empty:
            peer_slopes = peer_trends.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            peer_volatilities = peer_trends.std()
            correlations = peer_trends.corrwith(flagstar_trend)
        else:
            peer_slopes = pd.Series()
            peer_volatilities = pd.Series()
            correlations = pd.Series()

        overview = html.Div([
            html.Div([
                html.Div("Current Snapshot", className="stat-section-title"),
                html.Div([
                    html.Div("Average:", className="stat-label"),
                    html.Div(format_value(filtered_df[selected_metric].mean()))
                ], className="stat-row"),
                html.Div([
                    html.Div("Middle Value (Median):", className="stat-label"),
                    html.Div(format_value(filtered_df[selected_metric].median()))
                ], className="stat-row"),
                html.Div([
                    html.Div("Flagstar Bank's Value:", className="stat-label wf-highlight"),
                    html.Div(format_value(flagstar_value) if flagstar_value is not None else "N/A", className="wf-highlight")
                ], className="stat-row"),
                html.Div([
                    html.Div("Highest Value:", className="stat-label"),
                    html.Div(f"{format_value(filtered_df[selected_metric].max())} ({filtered_df.loc[filtered_df[selected_metric].idxmax(), 'Bank']})" if not filtered_df.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Lowest Value:", className="stat-label"),
                    html.Div(f"{format_value(filtered_df[selected_metric].min())} ({filtered_df.loc[filtered_df[selected_metric].idxmin(), 'Bank']})" if not filtered_df.empty else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),

            html.Div([
                html.Div("Flagstar Bank's Position", className="stat-section-title"),
                html.Div([
                    html.Div("Percentile Rank:", className="stat-label"),
                    html.Div(f"{wf_percentile:.1f}%" if wf_percentile is not None else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Flagstar Bank's Ranking:", className="stat-label"),
                    html.Div(f"#{wf_rank:.0f} out of {len(filtered_df)} banks" if wf_rank is not None else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Performance Group:", className="stat-label"),
                    html.Div(f"{'Top 25%' if flagstar_value > q3 else 'Bottom 25%' if flagstar_value <= q1 else 'Middle 50%'}" if flagstar_value is not None else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Standout Score:", className="stat-label"),
                    html.Div(f"{stats.zscore(filtered_df[selected_metric])[filtered_df['Bank'] == 'Flagstar Bank'].values[0]:.2f} " + 
                             "(How many standard deviations from the average)" if flagstar_value is not None else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),

            html.Div([
                html.Div(f"{trend_timeline}-Year Trend Analysis", className="stat-section-title"),
                html.Div([
                    html.Div("Flagstar Bank's Growth Rate:", className="stat-label"),
                    html.Div(f"{wf_slope:.4f} per year" if not np.isnan(wf_slope) else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Average Growth Rate:", className="stat-label"),
                    html.Div(f"{peer_slopes.mean():.4f} per year" if not peer_slopes.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Flagstar Bank's Stability:", className="stat-label"),
                    html.Div(f"{wf_volatility:.4f} (Lower means more stable)" if not np.isnan(wf_volatility) else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Average Stability:", className="stat-label"),
                    html.Div(f"{peer_volatilities.mean():.4f} (Lower means more stable)" if not peer_volatilities.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("How closely Flagstar Bank follows others:", className="stat-label"),
                    html.Div(f"{correlations.mean():.4f} (1 means perfect alignment, -1 means opposite movement)" if not correlations.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Bank moving most like Flagstar Bank:", className="stat-label"),
                    html.Div(f"{correlations.max():.4f} ({correlations.idxmax()})" if not correlations.empty else "N/A")
                ], className="stat-row"),
                html.Div([
                    html.Div("Bank moving least like Flagstar Bank:", className="stat-label"),
                    html.Div(f"{correlations.min():.4f} ({correlations.idxmin()})" if not correlations.empty else "N/A")
                ], className="stat-row"),
            ], className="stat-section"),
        ], style={"background-color": "#ffffff", "border-radius": "8px", "color": "#333333", "padding": "10px"})

        metric_definition = html.P(metric_definitions.get(selected_metric, ''), className="metric-definition")

        selected_peers_info = html.Div([
            html.P(f"Selected Peers: {len(selected_peers)} banks")
        ], style={"margin-top": "10px", "color": "#333333"})

        return fig, overview, metric_definition, selected_peers_info

    @app.callback(
        Output('bank-details', 'children'),
        Input('bar-chart', 'clickData'),
        Input('date-selector', 'value'),
        Input('metric-selector', 'value'),
        Input('individual-peer-selector', 'value')
    )
    def display_click_data(clickData, selected_date, selected_metric, selected_peers):
        selected_date = pd.to_datetime(selected_date).to_pydatetime()
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if clickData is None and triggered_id != 'date-selector':
            return html.P("Click on a bar to see details", style={"color": "#333333"})

        if triggered_id == 'date-selector' and clickData is None:
            return html.P("Select a date and click on a bar to see details", style={"color": "#333333"})

        if clickData:
            bank = clickData['points'][0]['x']
        else:
            bank = ctx.inputs['bar-chart.clickData']['points'][0]['x']

        selected_banks = ['Flagstar Bank'] + selected_peers

        if bank not in selected_banks:
            return html.P("Selected bank is not in the current comparison. Please select a displayed bank.", style={"color": "#333333"})

        bank_data = df[(df['Bank'] == bank) & (df['Date'] == selected_date)].iloc[0]

        def format_value(col, val):
            if col in dollar_format_metrics:
                return f"${val:,.0f}"
            else:
                return f"{val:.2f}"

        sorted_columns = [col for col in metric_order if col in bank_data.index] + [col for col in bank_data.index if col not in metric_order and col not in ['Bank', 'Date']]

        return [
            html.H5(f"{bank}", style={'color': '#D4AF37' if bank == 'Flagstar Bank' else '#333333'}),
            html.P(f"Date: {selected_date.strftime('%m/%d/%y')}", className="font-weight-bold", style={"color": "#333333"}),
            dbc.Table([
                html.Thead([
                    html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Definition")])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(col, style={"color": "#333333"}),
                        html.Td(format_value(col, bank_data[col]), style={"color": "#333333"}),
                        html.Td(metric_definitions.get(col, ''), style={'font-size': '0.8rem', "color": "#333333"})
                    ])
                    for col in sorted_columns
                ])
            ], striped=True, bordered=True, hover=True, size="sm", className="mt-3")
        ]

    return app, server

def main():
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    analyzer = BankDataAnalyzer()
    bank_info = [
        {"cert": "32541", "name": "Flagstar Bank, National Association"},
        {"cert": "6560", "name": "The Huntington National Bank"},
        {"cert": "12368", "name": "Regions Bank"},
        {"cert": "3511", "name": "Wells Fargo Bank, National Association"},
        {"cert": "3510", "name": "Bank of America, National Association"},
        {"cert": "7213", "name": "Citibank, National Association"},
        {"cert": "628", "name": "JPMorgan Chase Bank, National Association"},
        {"cert": "6548", "name": "U.S. Bank National Association"},
        {"cert": "6384", "name": "PNC Bank, National Association"},
        {"cert": "9846", "name": "Truist Bank"},
        {"cert": "33124", "name": "Goldman Sachs Bank USA"},
        {"cert": "32992", "name": "Morgan Stanley Bank, National Association"},
        {"cert": "18409", "name": "TD Bank, National Association"},
        {"cert": "4297", "name": "Capital One, National Association"},
        {"cert": "639", "name": "The Bank of New York Mellon"},
        {"cert": "6672", "name": "Fifth Third Bank, National Association"},
        {"cert": "57957", "name": "Citizens Bank, National Association"},
        {"cert": "57803", "name": "Ally Bank"},
        {"cert": "17534", "name": "KeyBank National Association"},
        {"cert": "5649", "name": "Discover Bank"},
        {"cert": "27314", "name": "Synchrony Bank"},
        {"cert": "29950", "name": "Santander Bank, N.A."}
    ]
    start_date = '20000331'
    end_date = '20240630'
    
    analyzer.fetch_data(bank_info, start_date, end_date)

    if not analyzer.institutions_data:
        print("No institution data was fetched. Exiting.")
        return None, None

    metrics_df = analyzer.calculate_metrics()

    sorted_columns = (
        ['Bank', 'Date'] +
        analyzer.dollar_format_metrics +
        [col for col in metrics_df.columns if col not in analyzer.dollar_format_metrics and col not in ['Bank', 'Date']]
    )
    metrics_df = metrics_df[sorted_columns]

    app, server = create_dashboard(metrics_df, analyzer.dollar_format_metrics, analyzer.metric_definitions, start_date, end_date)
    
    global df
    df = metrics_df.copy()
    
    return app, server

app, server = main()

if __name__ == "__main__":
    app.run_server(debug=False)
