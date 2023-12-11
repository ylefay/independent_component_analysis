import urllib.request
import pandas as pd
import os.path

horizon = "oneYear"
# Downloading the closing prices for the S&P500 index as well as the composing sectors
# spy_data = yf.download("^GSPC", start="2020-01-01", end="2023-10-22")["Close"]
sectors_urls = {
    'Information Technology': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=307',
    'Financials': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=279',
    'Health Care': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=253',
    'Consumer Discretionary': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=139',
    'Industrials': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=81',
    'Energy': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=25',
    'Consumer Staples': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=213',
    'Materials': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=41',
    'Utilities': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=356',
    'Telecommunication Services': f'https://www.spglobal.com/spdji/en/idsexport/file.xls?hostIdentifier=48190c8c-42c4-46af-8d1a-0cd5db894797&redesignExport=true&languageId=1&selectedModule=PerformanceGraphView&selectedSubModule=Graph&yearFlag={horizon}Flag&indexId=339'
    }
for sector in sectors_urls.keys():
    if not os.path.isfile('./data/sp500/' + sector + '.xls'):
        urllib.request.urlretrieve(sectors_urls[sector], './data/sp500/' + sector + '.xls')

# Concatenating the data into a single dataframe.
spy_data = None
for sector in sectors_urls.keys():
    data = pd.read_excel('./data/sp500/' + sector + '.xls', skiprows=4, index_col=0, parse_dates=True)
    data = data.dropna()[1:]
    data = data.rename(columns={'Unnamed: 1': sector})
    if spy_data is None:
        spy_data = data
    else:
        spy_data = pd.concat([spy_data, data], axis=1)

spy_data.to_csv('./data/sp500/sp500_per_sectors.csv')