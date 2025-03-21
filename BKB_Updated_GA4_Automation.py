#%%
import os
import numpy as np
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange, Dimension, Metric, RunReportRequest, OrderBy, Filter, FilterExpression, FilterExpressionList
)
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import gspread

# Set the path to your service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GA4_API.json'

# Initialize the GA4 client
client = BetaAnalyticsDataClient()

# Set your GA4 property ID
property_id = "465698830"

# Google Sheets API setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'google_sheets_api.json'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('sheets', 'v4', credentials=creds)
SPREADSHEET_ID = '1ycFh97caeOTf7nB8ht_SBsB2sxRv0vfdpvfAUtUE9Nk'
#%%
def create_combined_ga4_report():
    # Main GA4 metrics request
    main_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[Dimension(name="year"), Dimension(name="month")],
        metrics=[
            Metric(name="totalUsers"),
            Metric(name="activeUsers"),
            Metric(name="newUsers"),
            Metric(name="eventCount"),
            Metric(name="sessions")
        ],
        order_bys=[OrderBy(dimension={'dimension_name': 'month'})],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")]
    )

    # Traffic sources request
    traffic_source_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[
            OrderBy(dimension={'dimension_name': 'year'}, desc=True),
            OrderBy(dimension={'dimension_name': 'month'}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")]
    )

    # Send requests
    main_response = client.run_report(main_request)
    traffic_source_response = client.run_report(traffic_source_request)

    # Process main metrics
    def process_main_metrics(response):
        row_headers = [row.dimension_values for row in response.rows]
        metric_values = [row.metric_values for row in response.rows]

        data = []
        
        for i in range(len(row_headers)):
            year = int(row_headers[i][0].value)
            month = int(row_headers[i][1].value)
            total_users = float(metric_values[i][0].value)
            active_users = float(metric_values[i][1].value)
            new_users = float(metric_values[i][2].value)
            returning_users = total_users - new_users
            hits_events = float(metric_values[i][3].value)
            sessions = float(metric_values[i][4].value)

            data.append([year, month, total_users, active_users, returning_users, new_users, hits_events, sessions])

        df = pd.DataFrame(data, columns=[
            "Year", "Month", "Total Users", "Users/Active Users", "Returning Users", "New Users", "Hits/Events", "Sessions"
        ])

        return df

    # Process traffic sources
    def process_traffic_sources(response):
        data = {}
        for row in response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            source = row.dimension_values[2].value
            sessions = float(row.metric_values[0].value)
            
            key = f"{month:02}, {year}"
            if key not in data:
                data[key] = {"Organic Search": 0, "Direct": 0, "Referral": 0}
            
            if source.lower() == "google":
                data[key]["Organic Search"] += sessions
            elif source.lower() == "(direct)":
                data[key]["Direct"] += sessions
            elif source.lower() not in ["google", "(direct)"]:
                data[key]["Referral"] += sessions

        df = pd.DataFrame.from_dict(data, orient='index', columns=["Organic Search", "Direct", "Referral"])
        df.index.name = "Month-Year"
        df = df.reset_index()
        
        return df

    # Process both datasets
    main_df = process_main_metrics(main_response)
    traffic_sources_df = process_traffic_sources(traffic_source_response)

    # Combine datasets
    def combine_datasets(main_df, traffic_sources_df):
        # Create Month-Year column for both dataframes
        main_df['Month-Year'] = main_df['Month'].apply(lambda x: f'{x:02}') + ', ' + main_df['Year'].astype(str)
        
        # Merge dataframes
        combined_df = pd.merge(main_df, traffic_sources_df, on='Month-Year', how='left')
        
        # Reorder and select columns
        columns_order = [
            'Month-Year', 'Total Users', 'Users/Active Users', 'Returning Users', 
            'New Users', 'Hits/Events', 'Sessions', 
            'Organic Search', 'Direct', 'Referral'
        ]
        combined_df = combined_df[columns_order]
        
        # Create a datetime column for sorting
        combined_df['Sort_Date'] = pd.to_datetime(combined_df['Month-Year'], format='%m, %Y')
        
        # Sort in descending order (latest first)
        combined_df = combined_df.sort_values('Sort_Date', ascending=False)
        
        # Drop the sorting column
        combined_df = combined_df.drop(columns=['Sort_Date'])
        
        return combined_df

    return combine_datasets(main_df, traffic_sources_df)

def add_color_formatting(df):
    """
    Add color formatting based on trends and outliers
    Color coding:
    - Green: Above average (positive trend)
    - Red: Below average (negative trend)
    - Yellow: Slightly different from average
    """
    def get_color_class(column):
        # Calculate mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()
        
        def color_mapper(value):
            # More than 1 std dev above mean
            if value > mean + std:
                return 'positive-high-outlier'
            # Between 0.5 and 1 std dev above mean
            elif value > mean + (std/2):
                return 'positive-mild-outlier'
            # More than 1 std dev below mean
            elif value < mean - std:
                return 'negative-high-outlier'
            # Between 0.5 and 1 std dev below mean
            elif value < mean - (std/2):
                return 'negative-mild-outlier'
            # Close to average
            else:
                return 'average'
        
        return df[column].apply(color_mapper)
    
    # Columns to analyze (excluding Month-Year)
    numeric_columns = df.columns.drop('Month-Year').tolist()
    
    # Create color mapping for each column
    color_mapping = {col: get_color_class(col) for col in numeric_columns}
    
    return df, color_mapping

# Google Sheets API setup and export
def export_to_google_sheets(df, color_mapping):
    values = [df.columns.tolist()] + df.values.tolist()

    # Update the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range='MainDomain_Data!A1',
        valueInputOption='RAW',
        body={'values': values}
    )
    response = request.execute()

    # Formatting colors
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet('MainDomain_Data')

    batch_update_requests = [{
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet.id,
                        'startRowIndex': 1,  # Skip header row
                        'startColumnIndex': col_idx - 1,
                        'endColumnIndex': col_idx
                    }],
                    'gradientRule': {
                        'minpoint': {
                            'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                            'type': 'MIN'
                        },
                        'midpoint': {
                            'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                            'type': 'PERCENTILE',
                            'value': '50'
                        },
                        'maxpoint': {
                            'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                            'type': 'MAX'
                        }
                    }
                }
            }
        } for col_idx, col_name in enumerate(df.columns[1:], start=2)]  # Skip first column

    # Execute batch update
    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()
    print(f"{response.get('updatedCells')} cells updated.")

# Main execution
df = create_combined_ga4_report()
df_with_colors, color_mapping = add_color_formatting(df)
export_to_google_sheets(df_with_colors, color_mapping)

# Optional: Print the first few rows and color mapping
print(df_with_colors.head())
print("\nColor Mapping Legend:")
print("- Green shades: Performance above average (light to dark intensity)")
print("- Red shades: Performance below average (light to dark intensity)")
print("- White: Performance close to average")
# %%
# Get the specific sheet ID for 'MainDomain_Data'
spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
sheet_id = None
for sheet in spreadsheet['sheets']:
    if sheet['properties']['title'] == 'MainDomain_Data':
        sheet_id = sheet['properties']['sheetId']
        break

if sheet_id is None:
    raise ValueError("Sheet 'MainDomain_Data' not found")

# Calculate the y-axis range
max_value = df[['Total Users', 'Users/Active Users', 'Returning Users', 'New Users']].max().max()
min_value = df[['Total Users', 'Users/Active Users', 'Returning Users', 'New Users']].min().min()
y_axis_max = max_value * 1.1
y_axis_min = max(0, min_value * 0.9)

# First, delete any existing charts
delete_charts_request = {
    'requests': [{
        'deleteEmbeddedObject': {
            'objectId': chart['chartId']
        }
    } for chart in spreadsheet.get('sheets', [])[0].get('charts', [])]
}

if delete_charts_request['requests']:
    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=delete_charts_request
    ).execute()

# Define the new chart
chart = {
    'spec': {
        'title': 'User Metrics Over Time for All Domains (GA4 Data)',
        'basicChart': {
            'chartType': 'LINE',
            'legendPosition': 'RIGHT_LEGEND',
            'headerCount': 1,
            'axis': [
                {'position': 'BOTTOM_AXIS', 'title': 'Month-Year'},
                {
                    'position': 'LEFT_AXIS',
                    'title': 'Count',
                    'viewWindowOptions': {
                        'viewWindowMin': y_axis_min,
                        'viewWindowMax': y_axis_max
                    }
                }
            ],
            'domains': [{
                'domain': {
                    'sourceRange': {
                        'sources': [{
                            'sheetId': sheet_id,
                            'startRowIndex': 0,
                            'endRowIndex': df.shape[0] + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': 1
                        }]
                    }
                },
                'reversed': True
            }],
            'series': [
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 1,
                                'endColumnIndex': 2
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 0.4, 'green': 0.4, 'blue': 1.0},
                    'lineStyle': {'type': 'SOLID'}
                },
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 2,
                                'endColumnIndex': 3
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 1.0, 'green': 0.4, 'blue': 0.4},
                    'lineStyle': {'type': 'SOLID'}
                },
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 3,
                                'endColumnIndex': 4
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 1.0, 'green': 0.8, 'blue': 0.2},
                    'lineStyle': {'type': 'SOLID'}
                },
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 4,
                                'endColumnIndex': 5
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 0.2, 'green': 0.8, 'blue': 0.2},
                    'lineStyle': {'type': 'SOLID'}
                }
            ]
        }
    },
    'position': {
        'overlayPosition': {
            'anchorCell': {'sheetId': sheet_id, 'rowIndex': 0, 'columnIndex': 11},
            'widthPixels': 1200,
            'heightPixels': 600
        }
    }
}

# Add the new chart
chart_request = {
    'requests': [{
        'addChart': {
            'chart': chart
        }
    }]
}

response = service.spreadsheets().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body=chart_request
).execute()

print("Chart updated successfully in MainDomain_Data sheet.")
# %%
# Add Traffic Sources Chart
traffic_chart = {
    'spec': {
        'title': 'Traffic Sources Distribution Over Time',
        'basicChart': {
            'chartType': 'LINE',
            'legendPosition': 'RIGHT_LEGEND',
            'headerCount': 1,
            'axis': [
                {'position': 'BOTTOM_AXIS', 'title': 'Month-Year'},
                {
                    'position': 'LEFT_AXIS',
                    'title': 'Sessions'
                }
            ],
            'domains': [{
                'domain': {
                    'sourceRange': {
                        'sources': [{
                            'sheetId': sheet_id,
                            'startRowIndex': 0,
                            'endRowIndex': df.shape[0] + 1,
                            'startColumnIndex': 0,
                            'endColumnIndex': 1
                        }]
                    }
                },
                'reversed': True
            }],
            'series': [
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 7,
                                'endColumnIndex': 8
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 0.2, 'green': 0.6, 'blue': 1.0},
                    'lineStyle': {'type': 'SOLID'}
                },
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 8,
                                'endColumnIndex': 9
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 0.8, 'green': 0.4, 'blue': 0.0},
                    'lineStyle': {'type': 'SOLID'}
                },
                {
                    'series': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': df.shape[0] + 1,
                                'startColumnIndex': 9,
                                'endColumnIndex': 10
                            }]
                        }
                    },
                    'targetAxis': 'LEFT_AXIS',
                    'color': {'red': 0.6, 'green': 0.4, 'blue': 0.8},
                    'lineStyle': {'type': 'SOLID'}
                }
            ]
        }
    },
    'position': {
        'overlayPosition': {
            'anchorCell': {'sheetId': sheet_id, 'rowIndex': 30, 'columnIndex': 11},
            'widthPixels': 1200,
            'heightPixels': 600
        }
    }
}

# Add the traffic chart
traffic_chart_request = {
    'requests': [{
        'addChart': {
            'chart': traffic_chart
        }
    }]
}

response = service.spreadsheets().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body=traffic_chart_request
).execute()
# %%
def get_top_pages_overview():
    # First get top 20 pages overall with extended date range
    top_pages_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="pagePath"),
            Dimension(name="pageTitle")
        ],
        metrics=[
            Metric(name="screenPageViews")
        ],
        order_bys=[
            OrderBy(metric={"metric_name": "screenPageViews"}, desc=True)
        ],
        limit=100000,  # Increased to maximum limit
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")]
    )

    top_pages_response = client.run_report(top_pages_request)
    
    # In the first path mapping section, replace:
    path_mapping = {}
    for row in top_pages_response.rows:
        path = row.dimension_values[0].value
        normalized_path = path.rstrip('/')
        
        # Handle special cases and duplicates
        if path == "/" or path == "/home" or path == "/home/":
            normalized_path = "/"
        elif path.startswith("/glycan-search"):
            normalized_path = "/glycan-search/"
        elif path.startswith("/protein-search"):
            normalized_path = "/protein-search/"
            
        # Add views to get proper top pages
        views = int(row.metric_values[0].value)
        if normalized_path not in path_mapping:
            path_mapping[normalized_path] = {'path': path, 'views': views}
        else:
            path_mapping[normalized_path]['views'] += views

    # Sort by total views and get top 20
    consolidated_paths = [info['path'] for info in sorted(path_mapping.values(), 
                     key=lambda x: x['views'], reverse=True)][:20]


    # Then get monthly data with increased limits
    monthly_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="pagePath")
        ],
        metrics=[
            Metric(name="screenPageViews")
        ],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        limit=100000,  # Maximum limit to avoid data sampling
        offset=0  # Start from beginning
    )

    # Get all data by handling pagination
    all_rows = []
    while True:
        response = client.run_report(monthly_request)
        all_rows.extend(response.rows)
        
        if len(response.rows) < 100000:  # No more data to fetch
            break
            
        monthly_request.offset = len(all_rows)  # Update offset for next batch

    # Process into a dictionary with consolidated paths
    monthly_data = {}
    total_monthly_views = {}

    # In the monthly data processing section, modify:
    for row in all_rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        page_path = row.dimension_values[2].value
        views = int(row.metric_values[0].value)
        
        month_key = f"{month:02d}, {year}"
        
        if month_key not in monthly_data:
            monthly_data[month_key] = {path: 0 for path in consolidated_paths}
            total_monthly_views[month_key] = 0
        
        # Normalize path and add views
        normalized_path = page_path.rstrip('/')
        if normalized_path == "/" or normalized_path == "/home":
            normalized_path = "/"
            monthly_data[month_key]["/"] += views
        elif normalized_path.startswith("/glycan-search"):
            monthly_data[month_key]["/glycan-search/"] += views
        elif normalized_path.startswith("/protein-search"):
            monthly_data[month_key]["/protein-search/"] += views
        elif page_path in consolidated_paths:
            monthly_data[month_key][page_path] += views
        
        total_monthly_views[month_key] += views


    # Create DataFrame and continue with existing code...
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df['Total Pageviews'] = pd.Series(total_monthly_views)
    
    cols = ['Total Pageviews'] + [col for col in df.columns if col != 'Total Pageviews']
    df = df[cols]
    
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date (latest first)
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    gc = gspread.authorize(creds)
    
    sheet_title = 'MainDomain_Top20Pages'
    
    # Convert DataFrame to values
    values = [df.columns.tolist()] + df.values.tolist()
    
    # Update sheet
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    sheet.clear()
    sheet.update('A1', values)

    # Apply conditional formatting to all numeric columns
    format_request = {
        'requests': [{
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet.id,
                        'startRowIndex': 1,
                        'startColumnIndex': col_idx,
                        'endColumnIndex': col_idx + 1
                    }],
                    'gradientRule': {
                        'minpoint': {'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404}, 'type': 'MIN'},
                        'midpoint': {'color': {'red': 1, 'green': 1, 'blue': 1}, 'type': 'PERCENTILE', 'value': '50'},
                        'maxpoint': {'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420}, 'type': 'MAX'}
                    }
                }
            }
        } for col_idx in range(1, len(df.columns))]
    }
    
    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body=format_request
    ).execute()

    print("Top 20 pages report created successfully.")

# Execute the function
get_top_pages_overview()

# %%
def add_top_pages_chart():
    SHEET_TITLE = 'MainDomain_Top20Pages'
    # Get sheet ID
    spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    sheet_id = None
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == SHEET_TITLE:
            sheet_id = sheet['properties']['sheetId']
            break
            
    if sheet_id is None:
        raise ValueError(f"Sheet '{SHEET_TITLE}' not found")

    # Get the data to determine dimensions
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TITLE)
    data = worksheet.get_all_values()
    num_rows = len(data)
    num_columns = len(data[0])

    padding_request = {
        'requests': [{
            'updateDimensionProperties': {
                'range': {
                    'sheetId': sheet_id,
                    'dimension': 'COLUMNS',
                    'startIndex': num_columns,
                    'endIndex': num_columns + 5
                },
                'properties': {
                    'pixelSize': 75
                },
                'fields': 'pixelSize'
            }
        }]
    }

    # Define colors for the lines
    colors = [
        {'red': 0.4, 'green': 0.4, 'blue': 1.0},  # Blue
        {'red': 1.0, 'green': 0.4, 'blue': 0.4},  # Red
        {'red': 0.4, 'green': 1.0, 'blue': 0.4},  # Green
        {'red': 1.0, 'green': 0.8, 'blue': 0.2},  # Yellow
        {'red': 0.8, 'green': 0.4, 'blue': 0.8},  # Purple
        {'red': 0.4, 'green': 0.8, 'blue': 1.0},  # Light Blue
        {'red': 1.0, 'green': 0.6, 'blue': 0.4},  # Orange
        {'red': 0.6, 'green': 0.4, 'blue': 0.2},  # Brown
        {'red': 0.8, 'green': 0.8, 'blue': 0.4},  # Light Yellow
        {'red': 0.4, 'green': 0.8, 'blue': 0.6}   # Teal
    ]

    # Create chart specification
    chart = {
        'spec': {
            'title': 'Top Pages Views Over Time',
            'basicChart': {
                'chartType': 'LINE',
                'legendPosition': 'RIGHT_LEGEND',
                'headerCount': 1,
                'axis': [
                    {
                        'position': 'BOTTOM_AXIS',
                        'title': 'Month-Year'
                    },
                    {
                        'position': 'LEFT_AXIS',
                        'title': 'Page Views'
                    }
                ],
                'domains': [{
                    'domain': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': num_rows,
                                'startColumnIndex': 0,
                                'endColumnIndex': 1
                            }]
                        }
                    },
                    'reversed': True
                }],
                'series': []
            }
        },
        'position': {
            'overlayPosition': {
                'anchorCell': {
                    'sheetId': sheet_id,
                    'rowIndex': 0,
                    'columnIndex': num_columns + 3
                },
                'widthPixels': 900,
                'heightPixels': 520
            }
        }
    }

    # Add series for top 10 pages (columns 1 to 11, including Total Pageviews)
    for idx in range(1, 11):
        series = {
            'series': {
                'sourceRange': {
                    'sources': [{
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': num_rows,
                        'startColumnIndex': idx,
                        'endColumnIndex': idx + 1
                    }]
                }
            },
            'targetAxis': 'LEFT_AXIS',
            'color': colors[idx - 1] if idx - 1 < len(colors) else {'red': 0.5, 'green': 0.5, 'blue': 0.5},
            'lineStyle': {'type': 'SOLID', 'width': 2}
        }
        chart['spec']['basicChart']['series'].append(series)

    # Create the chart request
    chart_request = {
        'requests': [{
            'addChart': {
                'chart': chart
            }
        }]
    }

    # Execute the request
    try:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=chart_request
        ).execute()
        print("Chart added successfully.")
    except Exception as e:
        print(f"Error creating chart: {str(e)}")

# Execute the function
add_top_pages_chart()
# %%
## Bottom 10 Pages: MainDomain_Bottom10Pages
def create_bottom_pages_trend_report():

    # First request: Get overall bottom 10 pages
    bottom_pages_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="pagePath"),
            Dimension(name="pageTitle")
        ],
        metrics=[
            Metric(name="screenPageViews")
        ],
        order_bys=[
            OrderBy(metric={"metric_name": "screenPageViews"}, desc=False)  # Changed to False for bottom pages
        ],
        limit=10,
        date_ranges=[DateRange(start_date="2023-12-01", end_date="today")]
    )

    bottom_pages_response = client.run_report(bottom_pages_request)
    bottom_pages = [(row.dimension_values[0].value, row.dimension_values[1].value) 
                   for row in bottom_pages_response.rows]

    # Second request: Get monthly data with explicit date range
    monthly_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="pagePath")
        ],
        metrics=[
            Metric(name="screenPageViews")
        ],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-12-01", end_date="today")],
        limit=50000
    )

    monthly_response = client.run_report(monthly_request)

    # Process monthly data with explicit date range handling
    monthly_data = {}
    
    # Create all month-year combinations from 2020 to today
    current_date = pd.Timestamp.now()
    start_date = pd.Timestamp('2023-12-01')
    date_range = pd.date_range(start=start_date, end=current_date, freq='M')
    
    # Initialize the dictionary with all possible months
    for date in date_range:
        month_key = f"{date.month:02d}, {date.year}"
        monthly_data[month_key] = {page[0]: 0 for page in bottom_pages}

    # Fill in the actual data
    for row in monthly_response.rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        page_path = row.dimension_values[2].value
        views = int(row.metric_values[0].value)
        
        month_key = f"{month:02d}, {year}"
        if month_key in monthly_data and page_path in monthly_data[month_key]:
            monthly_data[month_key][page_path] = views

    # Create DataFrame with all months
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date (latest first)
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Format column headers with page titles
    header_mapping = {page[0]: f"{page[1]}\n({page[0]})" for page in bottom_pages}
    df = df.rename(columns=header_mapping)

    sheet_title = 'MainDomain_Bottom10Pages'  # Changed sheet title

    # Convert DataFrame to values
    values = [df.columns.tolist()] + df.values.tolist()

    # Update the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f'{sheet_title}!A1',
        valueInputOption='RAW',
        body={'values': values}
    )
    response = request.execute()

    # Format header
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    
    format_requests = [{
        'repeatCell': {
            'range': {
                'sheetId': sheet.id,
                'startRowIndex': 0,
                'endRowIndex': 1
            },
            'cell': {
                'userEnteredFormat': {
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                    'textFormat': {'bold': True},
                    'wrapStrategy': 'WRAP',
                    'verticalAlignment': 'MIDDLE',
                    'horizontalAlignment': 'CENTER'
                }
            },
            'fields': 'userEnteredFormat(backgroundColor,textFormat,wrapStrategy,verticalAlignment,horizontalAlignment)'
        }
    }]

    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={'requests': format_requests}
    ).execute()

    print("Bottom pages trend report created successfully.")

# Execute the function
create_bottom_pages_trend_report()

#%%
## Top 10 Referrals
def add_top_referrals_chart(df, service, spreadsheet_id, sheet_title):
    # Get sheet ID
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_title:
            sheet_id = sheet['properties']['sheetId']
            break
            
    if sheet_id is None:
        raise ValueError(f"Sheet '{sheet_title}' not found")

    # Get the number of columns
    num_columns = len(df.columns)

    # Create the chart
    chart = {
        'spec': {
            'title': 'Top Referral Sources Over Time',
            'basicChart': {
                'chartType': 'LINE',
                'legendPosition': 'RIGHT_LEGEND',
                'headerCount': 1,
                'axis': [
                    {'position': 'BOTTOM_AXIS', 'title': 'Month-Year'},
                    {
                        'position': 'LEFT_AXIS',
                        'title': 'Sessions'
                    }
                ],
                'domains': [{
                    'domain': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': len(df) + 1,
                                'startColumnIndex': 0,
                                'endColumnIndex': 1
                            }]
                        }
                    },
                    'reversed': True
                }],
                'series': []
            }
        },
        'position': {
            'overlayPosition': {
                'anchorCell': {
                    'sheetId': sheet_id,
                    'rowIndex': 0,
                    'columnIndex': num_columns + 2
                },
                'widthPixels': 1200,
                'heightPixels': 600
            }
        }
    }

    # Colors for different referral sources
    colors = [
        {'red': 0.4, 'green': 0.4, 'blue': 1.0},  # Blue
        {'red': 1.0, 'green': 0.4, 'blue': 0.4},  # Red
        {'red': 0.4, 'green': 0.8, 'blue': 0.4},  # Green
        {'red': 1.0, 'green': 0.8, 'blue': 0.2},  # Yellow
        {'red': 0.8, 'green': 0.4, 'blue': 0.8},  # Purple
        {'red': 0.4, 'green': 0.8, 'blue': 1.0},  # Light Blue
        {'red': 1.0, 'green': 0.6, 'blue': 0.4},  # Orange
        {'red': 0.6, 'green': 0.4, 'blue': 0.2},  # Brown
        {'red': 0.8, 'green': 0.8, 'blue': 0.4},  # Light Yellow
        {'red': 0.4, 'green': 0.8, 'blue': 0.6}   # Teal
    ]

    # Add series for each referral source
    for idx, col in enumerate(df.columns[1:], start=1):
        series = {
            'series': {
                'sourceRange': {
                    'sources': [{
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': len(df) + 1,
                        'startColumnIndex': idx,
                        'endColumnIndex': idx + 1
                    }]
                }
            },
            'targetAxis': 'LEFT_AXIS',
            'color': colors[idx - 1] if idx - 1 < len(colors) else colors[-1],
            'lineStyle': {'type': 'SOLID', 'width': 2}
        }
        chart['spec']['basicChart']['series'].append(series)

    # Add the chart
    chart_request = {
        'requests': [{
            'addChart': {
                'chart': chart
            }
        }]
    }

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=chart_request
    ).execute()
def create_top_referrals_trend_report():

    # First request: Get overall top 10 referral sources
    top_referrals_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="sessionSource"),
            Dimension(name="sessionMedium")
        ],
        metrics=[
            Metric(name="sessions")
        ],
        order_bys=[
            OrderBy(metric={"metric_name": "sessions"}, desc=True)
        ],
        limit=10,
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        dimension_filter={
            'filter': {
                'field_name': 'sessionMedium',
                'string_filter': {
                    'value': 'referral',
                    'match_type': 'EXACT'
                }
            }
        }
    )

    top_referrals_response = client.run_report(top_referrals_request)
    top_referrals = [(row.dimension_values[0].value, row.dimension_values[1].value) 
                     for row in top_referrals_response.rows]

    # Second request: Get monthly data for these referral sources
    monthly_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[
            Metric(name="sessions")
        ],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        limit=50000
    )

    monthly_response = client.run_report(monthly_request)

    # Process monthly data
    monthly_data = {}
    
    # Create all month-year combinations
    current_date = pd.Timestamp.now()
    start_date = pd.Timestamp('2020-01-01')
    date_range = pd.date_range(start=start_date, end=current_date, freq='M')
    
    # Initialize the dictionary with all possible months
    for date in date_range:
        month_key = f"{date.month:02d}, {date.year}"
        monthly_data[month_key] = {referral[0]: 0 for referral in top_referrals}

    # Fill in the actual data
    for row in monthly_response.rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        source = row.dimension_values[2].value
        sessions = int(row.metric_values[0].value)
        
        month_key = f"{month:02d}, {year}"
        if month_key in monthly_data and source in monthly_data[month_key]:
            monthly_data[month_key][source] = sessions

    # Create DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date (latest first)
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Filter out rows where all numeric columns are 0
    numeric_columns = df.columns.drop('Month-Year')
    df = df[~(df[numeric_columns] == 0).all(axis=1)]

    sheet_title = 'MainDomain_Top10Referrals'

    # Convert DataFrame to values
    values = [df.columns.tolist()] + df.values.tolist()

    # Update the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f'{sheet_title}!A1',
        valueInputOption='RAW',
        body={'values': values}
    )
    response = request.execute()

    # Format header
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    
    format_requests = [
        {
            'repeatCell': {
                'range': {
                    'sheetId': sheet.id,
                    'startRowIndex': 0,
                    'endRowIndex': 1
                },
                'cell': {
                    'userEnteredFormat': {
                        'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                        'textFormat': {'bold': True},
                        'wrapStrategy': 'WRAP',
                        'verticalAlignment': 'MIDDLE',
                        'horizontalAlignment': 'CENTER'
                    }
                },
                'fields': 'userEnteredFormat(backgroundColor,textFormat,wrapStrategy,verticalAlignment,horizontalAlignment)'
            }
        }
    ]

    # Apply conditional formatting to all numeric columns
    for col_idx in range(1, len(df.columns)):
        format_requests.append({
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet.id,
                        'startRowIndex': 1,
                        'startColumnIndex': col_idx,
                        'endColumnIndex': col_idx + 1
                    }],
                    'gradientRule': {
                        'minpoint': {'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404}, 'type': 'MIN'},
                        'midpoint': {'color': {'red': 1, 'green': 1, 'blue': 1}, 'type': 'PERCENTILE', 'value': '50'},
                        'maxpoint': {'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420}, 'type': 'MAX'}
                    }
                },
                'index': 0
            }
        })

    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={'requests': format_requests}
    ).execute()

    # Add the chart
    add_top_referrals_chart(df, service, SPREADSHEET_ID, sheet_title)

    print("Top referrals trend report created successfully.")

# Execute the function
create_top_referrals_trend_report()
#%% 
## Top 10 Countries Monthly
def create_top_countries_report():
    try:
        # First get top 10 countries overall
        top_countries_request = RunReportRequest(
            property='properties/' + property_id,
            dimensions=[
                Dimension(name="country")
            ],
            metrics=[
                Metric(name="engagedSessions")
            ],
            order_bys=[
                OrderBy(metric={"metric_name": "engagedSessions"}, desc=True)
            ],
            limit=10,
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")]
        )

        top_countries_response = client.run_report(top_countries_request)
        top_countries = [row.dimension_values[0].value for row in top_countries_response.rows]

        # Then get monthly data for these countries
        monthly_request = RunReportRequest(
            property='properties/' + property_id,
            dimensions=[
                Dimension(name="year"),
                Dimension(name="month"),
                Dimension(name="country")
            ],
            metrics=[
                Metric(name="engagedSessions")
            ],
            order_bys=[
                OrderBy(dimension={"dimension_name": "year"}, desc=True),
                OrderBy(dimension={"dimension_name": "month"}, desc=True)
            ],
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")]
        )

        monthly_response = client.run_report(monthly_request)

        # Process into a dictionary with flattened structure
        monthly_data = {}
        total_monthly_sessions = {}
        
        for row in monthly_response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            country = row.dimension_values[2].value
            
            if country in top_countries:
                month_key = f"{month:02d}, {year}"
                if month_key not in monthly_data:
                    monthly_data[month_key] = {country: 0 for country in top_countries}
                    total_monthly_sessions[month_key] = 0
                
                sessions = int(row.metric_values[0].value)
                monthly_data[month_key][country] = sessions
                total_monthly_sessions[month_key] += sessions

        # Create DataFrame
        df = pd.DataFrame.from_dict(monthly_data, orient='index')
        df['Total Engaged Sessions'] = pd.Series(total_monthly_sessions)
        df.index.name = 'Month-Year'
        df = df.reset_index()

        # Sort by date
        df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
        df = df.sort_values('Sort_Date', ascending=False)
        df = df.drop('Sort_Date', axis=1)

        # Reorder columns to put Total first
        cols = ['Month-Year', 'Total Engaged Sessions'] + [col for col in df.columns if col not in ['Month-Year', 'Total Engaged Sessions']]
        df = df[cols]

        gc = gspread.authorize(creds)
        
        sheet_title = 'MainDomain_Top10Countries_Monthly'

        # Create or get worksheet
        try:
            worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
        except:
            worksheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(sheet_title, rows=100, cols=20)

        # Clear existing content
        worksheet.clear()

        # Update values
        values = [df.columns.tolist()] + df.values.tolist()
        worksheet.update('A1', values, value_input_option='RAW')

        # Apply formatting
        format_requests = []
        
        # Add header formatting
        format_requests.append({
            'repeatCell': {
                'range': {
                    'sheetId': worksheet.id,
                    'startRowIndex': 0,
                    'endRowIndex': 1
                },
                'cell': {
                    'userEnteredFormat': {
                        'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                        'textFormat': {'bold': True},
                        'horizontalAlignment': 'CENTER'
                    }
                },
                'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
            }
        })

        # Add gradient conditional formatting for numeric columns
        for col_idx in range(1, len(df.columns)):
            format_requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [{
                            'sheetId': worksheet.id,
                            'startRowIndex': 1,
                            'startColumnIndex': col_idx,
                            'endColumnIndex': col_idx + 1
                        }],
                        'gradientRule': {
                            'minpoint': {
                                'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},
                                'type': 'MIN'
                            },
                            'midpoint': {
                                'color': {'red': 1, 'green': 1, 'blue': 1},
                                'type': 'PERCENTILE',
                                'value': '50'
                            },
                            'maxpoint': {
                                'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},
                                'type': 'MAX'
                            }
                        }
                    }
                }
            })

        # Execute formatting requests
        format_request = {'requests': format_requests}
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=format_request
        ).execute()

        colors = [
            {'red': 0.4, 'green': 0.4, 'blue': 1.0},  # Blue
            {'red': 1.0, 'green': 0.4, 'blue': 0.4},  # Red
            {'red': 0.4, 'green': 1.0, 'blue': 0.4},  # Green
            {'red': 1.0, 'green': 0.8, 'blue': 0.2},  # Yellow
            {'red': 0.8, 'green': 0.4, 'blue': 0.8},  # Purple
            {'red': 0.4, 'green': 0.8, 'blue': 1.0},  # Light Blue
            {'red': 1.0, 'green': 0.6, 'blue': 0.4},  # Orange
            {'red': 0.6, 'green': 0.4, 'blue': 0.2},  # Brown
            {'red': 0.8, 'green': 0.8, 'blue': 0.4},  # Light Yellow
            {'red': 0.4, 'green': 0.8, 'blue': 0.6}   # Teal
        ]

        # Add chart
        chart_request = {
            'requests': [{
                'addChart': {
                    'chart': {
                        'spec': {
                            'title': 'Top Countries Engagement Over Time',
                            'basicChart': {
                                'chartType': 'LINE',
                                'legendPosition': 'RIGHT_LEGEND',
                                'headerCount': 1,
                                'axis': [
                                    {
                                        'position': 'BOTTOM_AXIS',
                                        'title': 'Month-Year'
                                    },
                                    {
                                        'position': 'LEFT_AXIS',
                                        'title': 'Engaged Sessions'
                                    }
                                ],
                                'domains': [{
                                    'domain': {
                                        'sourceRange': {
                                            'sources': [{
                                                'sheetId': worksheet.id,
                                                'startRowIndex': 0,
                                                'endRowIndex': len(values),
                                                'startColumnIndex': 0,
                                                'endColumnIndex': 1
                                            }]
                                        }
                                    },
                                    'reversed': True  # This will show old to new dates
                                }],
                                'series': [
                                    {
                                        'series': {
                                            'sourceRange': {
                                                'sources': [{
                                                    'sheetId': worksheet.id,
                                                    'startRowIndex': 0,
                                                    'endRowIndex': len(values),
                                                    'startColumnIndex': idx + 1,
                                                    'endColumnIndex': idx + 2
                                                }]
                                            }
                                        },
                                        'targetAxis': 'LEFT_AXIS',
                                        'lineStyle': {'width': 2},
                                        'color': colors[idx % len(colors)]  # Add different colors for each line
                                    } for idx in range(len(top_countries))
                                ]
                            }
                        },
                        'position': {
                            'overlayPosition': {
                                'anchorCell': {
                                    'sheetId': worksheet.id,
                                    'rowIndex': 0,
                                    'columnIndex': len(df.columns) + 2  # Position chart after the table with padding
                                },
                                'widthPixels': 1200,
                                'heightPixels': 600
                            }
                        }
                    }
                }
            }]
        }

        # Execute chart request
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=chart_request
        ).execute()


        print("Top 10 countries report created successfully.")
    except Exception as e:
        print(f"Error creating report: {str(e)}")

# Execute the function
create_top_countries_report()
#%%
#Subdomains Overview
def explore_subdomains():

    # First get all hostnames/subdomains
    hostname_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="hostname")
        ],
        metrics=[
            Metric(name="screenPageViews"),
            Metric(name="engagedSessions"),
            Metric(name="totalUsers")
        ],
        order_bys=[
            OrderBy(metric={"metric_name": "screenPageViews"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")]
    )

    hostname_response = client.run_report(hostname_request)

    # Process into DataFrame
    data = []
    for row in hostname_response.rows:
        hostname = row.dimension_values[0].value
        pageviews = int(row.metric_values[0].value)
        sessions = int(row.metric_values[1].value)
        users = int(row.metric_values[2].value)
        data.append([hostname, pageviews, sessions, users])

    df = pd.DataFrame(data, columns=["Hostname", "Pageviews", "Engaged Sessions", "Users"])

    # Print the list of all subdomains
    print("List of all subdomains:")
    print(df["Hostname"].tolist())

    gc = gspread.authorize(creds)
    
    sheet_title = 'Subdomains_Overview'

    # Create or get worksheet
    try:
        worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
        worksheet.clear()
    except:
        worksheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(sheet_title, rows=100, cols=20)

    # Update values
    values = [df.columns.tolist()] + df.values.tolist()
    worksheet.update(values=values, range_name='A1')

    # Format header
    format_requests = [{
        'repeatCell': {
            'range': {
                'sheetId': worksheet.id,
                'startRowIndex': 0,
                'endRowIndex': 1
            },
            'cell': {
                'userEnteredFormat': {
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                    'textFormat': {'bold': True},
                    'horizontalAlignment': 'CENTER'
                }
            },
            'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)'
        }
    }]

    # Add conditional formatting for numeric columns
    for col_idx in range(1, 4):  # Columns B, C, D
        format_requests.append({
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': worksheet.id,
                        'startRowIndex': 1,
                        'startColumnIndex': col_idx,
                        'endColumnIndex': col_idx + 1
                    }],
                    'gradientRule': {
                        'minpoint': {
                            'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},
                            'type': 'MIN'
                        },
                        'midpoint': {
                            'color': {'red': 1, 'green': 1, 'blue': 1},
                            'type': 'PERCENTILE',
                            'value': '50'
                        },
                        'maxpoint': {
                            'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},
                            'type': 'MAX'
                        }
                    }
                }
            }
        })

    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={'requests': format_requests}
    ).execute()

    print("Subdomains overview created successfully.")
    return df  # Return DataFrame for further analysis

# Execute the function
subdomains_df = explore_subdomains()
print("\nAvailable subdomains:")
print(subdomains_df[['Hostname', 'Pageviews']].to_string())
# %%
#Subdomain Trend Report - biomarkerkb.org and www.biomarkerkb.org
def create_biomarkerkbportal_ga4_report():
    # Main GA4 metrics request
    main_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[Dimension(name="year"), Dimension(name="month")],
        metrics=[
            Metric(name="totalUsers"),
            Metric(name="activeUsers"),
            Metric(name="newUsers"),
            Metric(name="eventCount"),
            Metric(name="sessions")
        ],
        order_bys=[OrderBy(dimension={'dimension_name': 'month'})],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        dimension_filter={
        'filter': {
            'field_name': 'hostname',
            'string_filter': {
                'value': 'biomarkerkb.org',
                'match_type': 'EXACT'
            },
            'in_list_filter': {
                'values': ['biomarkerkb.org', 'www.biomarkerkb.org']
            }
        }
    }
    )

    # Traffic sources request
    traffic_source_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[
            OrderBy(dimension={'dimension_name': 'year'}, desc=True),
            OrderBy(dimension={'dimension_name': 'month'}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        dimension_filter={
        'filter': {
            'field_name': 'hostname',
            'string_filter': {
                'value': 'biomarkerkb.org',
                'match_type': 'EXACT'
            },
            'in_list_filter': {
                'values': ['biomarkerkb.org', 'www.biomarkerkb.org']
            }
        }
    }
    )

    # Send requests
    main_response = client.run_report(main_request)
    traffic_source_response = client.run_report(traffic_source_request)

    # Process main metrics
    def process_glgenportal_metrics(response):
        row_headers = [row.dimension_values for row in response.rows]
        metric_values = [row.metric_values for row in response.rows]

        data = []
        
        for i in range(len(row_headers)):
            year = int(row_headers[i][0].value)
            month = int(row_headers[i][1].value)
            total_users = float(metric_values[i][0].value)
            active_users = float(metric_values[i][1].value)
            new_users = float(metric_values[i][2].value)
            returning_users = total_users - new_users
            hits_events = float(metric_values[i][3].value)
            sessions = float(metric_values[i][4].value)

            data.append([year, month, total_users, active_users, returning_users, new_users, hits_events, sessions])

        df = pd.DataFrame(data, columns=[
            "Year", "Month", "Total Users", "Users/Active Users", "Returning Users", "New Users", "Hits/Events", "Sessions"
        ])

        return df

    # Process traffic sources
    def process_glgenportal_traffic_sources(response):
        data = {}
        for row in response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            source = row.dimension_values[2].value
            sessions = float(row.metric_values[0].value)
            
            key = f"{month:02}, {year}"
            if key not in data:
                data[key] = {"Organic Search": 0, "Direct": 0, "Referral": 0}
            
            if source.lower() == "google":
                data[key]["Organic Search"] += sessions
            elif source.lower() == "(direct)":
                data[key]["Direct"] += sessions
            elif source.lower() not in ["google", "(direct)"]:
                data[key]["Referral"] += sessions

        df = pd.DataFrame.from_dict(data, orient='index', columns=["Organic Search", "Direct", "Referral"])
        df.index.name = "Month-Year"
        df = df.reset_index()
        
        return df

    # Process both datasets
    main_df = process_glgenportal_metrics(main_response)
    traffic_sources_df = process_glgenportal_traffic_sources(traffic_source_response)

    # Combine datasets
    def combine_datasets(main_df, traffic_sources_df):
        # Create Month-Year column for both dataframes
        main_df['Month-Year'] = main_df['Month'].apply(lambda x: f'{x:02}') + ', ' + main_df['Year'].astype(str)
        
        # Merge dataframes
        combined_df = pd.merge(main_df, traffic_sources_df, on='Month-Year', how='left')
        
        # Reorder and select columns
        columns_order = [
            'Month-Year', 'Total Users', 'Users/Active Users', 'Returning Users', 
            'New Users', 'Hits/Events', 'Sessions', 
            'Organic Search', 'Direct', 'Referral'
        ]
        combined_df = combined_df[columns_order]
        
        # Create a datetime column for sorting
        combined_df['Sort_Date'] = pd.to_datetime(combined_df['Month-Year'], format='%m, %Y')
        
        # Sort in descending order (latest first)
        combined_df = combined_df.sort_values('Sort_Date', ascending=False)
        
        # Drop the sorting column
        combined_df = combined_df.drop(columns=['Sort_Date'])
        
        return combined_df

    return combine_datasets(main_df, traffic_sources_df)

def add_color_formatting(df):
    """
    Add color formatting based on trends and outliers
    Color coding:
    - Green: Above average (positive trend)
    - Red: Below average (negative trend)
    - Yellow: Slightly different from average
    """
    def get_color_class(column):
        # Calculate mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()
        
        def color_mapper(value):
            # More than 1 std dev above mean
            if value > mean + std:
                return 'positive-high-outlier'
            # Between 0.5 and 1 std dev above mean
            elif value > mean + (std/2):
                return 'positive-mild-outlier'
            # More than 1 std dev below mean
            elif value < mean - std:
                return 'negative-high-outlier'
            # Between 0.5 and 1 std dev below mean
            elif value < mean - (std/2):
                return 'negative-mild-outlier'
            # Close to average
            else:
                return 'average'
        
        return df[column].apply(color_mapper)
    
    # Columns to analyze (excluding Month-Year)
    numeric_columns = df.columns.drop('Month-Year').tolist()
    
    # Create color mapping for each column
    color_mapping = {col: get_color_class(col) for col in numeric_columns}
    
    return df, color_mapping

# Google Sheets API setup and export
def export_to_google_sheets(df, color_mapping):
    # Convert DataFrame to list of lists for Google Sheets
    values = [df.columns.tolist()] + df.values.tolist()

    # Update the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range='BiomarkerKB_Portal_Overview!A1',
        valueInputOption='RAW',
        body={'values': values}
    )
    response = request.execute()

    # Formatting colors
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet('BiomarkerKB_Portal_Overview')

    batch_update_requests = [{
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet.id,
                        'startRowIndex': 1,  # Skip header row
                        'startColumnIndex': col_idx - 1,
                        'endColumnIndex': col_idx
                    }],
                    'gradientRule': {
                        'minpoint': {
                            'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                            'type': 'MIN'
                        },
                        'midpoint': {
                            'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                            'type': 'PERCENTILE',
                            'value': '50'
                        },
                        'maxpoint': {
                            'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                            'type': 'MAX'
                        }
                    }
                }
            }
        } for col_idx, col_name in enumerate(df.columns[1:], start=2)]  # Skip first column

    # Execute batch update
    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()
    print(f"{response.get('updatedCells')} cells updated.")

# Main execution
df = create_biomarkerkbportal_ga4_report()
df_with_colors, color_mapping = add_color_formatting(df)
export_to_google_sheets(df_with_colors, color_mapping)

# Optional: Print the first few rows and color mapping
print(df_with_colors.head())
print("\nColor Mapping Legend:")
print("- Green shades: Performance above average (light to dark intensity)")
print("- Red shades: Performance below average (light to dark intensity)")
print("- White: Performance close to average")
#%%

#%%
#BiomarkerKB Top Pages Overview
from google.analytics.data_v1beta.types import (
    DateRange, Dimension, Metric, RunReportRequest, OrderBy, Filter, FilterExpression, FilterExpressionList
)


def get_biomarkerkb_top_pages_overview():

    # Define filter for BiomarkerKB Portal (Exact match for www.biomarkerkb.org and biomarkerkb.org)
    subdomain_filter = FilterExpression(
        or_group=FilterExpressionList(
            expressions=[
                FilterExpression(
                    filter=Filter(
                        field_name="hostname",
                        string_filter=Filter.StringFilter(
                            value="biomarkerkb.org",
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                ),
                FilterExpression(
                    filter=Filter(
                        field_name="hostname",
                        string_filter=Filter.StringFilter(
                            value="www.biomarkerkb.org",
                            match_type=Filter.StringFilter.MatchType.EXACT
                        )
                    )
                )
            ]
        )
    )

    # First request: Get top 20 pages overall with extended date range (filtered for BiomarkerKB)
    top_pages_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="pagePath"),
            Dimension(name="pageTitle")
        ],
        metrics=[Metric(name="screenPageViews")],
        order_bys=[OrderBy(metric={"metric_name": "screenPageViews"}, desc=True)],
        limit=100000,
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    top_pages_response = client.run_report(top_pages_request)

    # Process path mapping
    path_mapping = {}
    for row in top_pages_response.rows:
        path = row.dimension_values[0].value
        normalized_path = path.rstrip('/')

        # Handle special cases and duplicates
        if path in ["/", "/home", "/home/"]:
            normalized_path = "/"

        # Add views to get proper top pages
        views = int(row.metric_values[0].value)
        if normalized_path not in path_mapping:
            path_mapping[normalized_path] = {'path': path, 'views': views}
        else:
            path_mapping[normalized_path]['views'] += views

    # Sort by total views and get top 20
    consolidated_paths = [info['path'] for info in sorted(path_mapping.values(), key=lambda x: x['views'], reverse=True)][:20]

    # Second request: Get monthly data for these top pages
    monthly_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="pagePath")
        ],
        metrics=[Metric(name="screenPageViews")],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        limit=100000,
        offset=0,
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    # Get all data by handling pagination
    all_rows = []
    while True:
        response = client.run_report(monthly_request)
        all_rows.extend(response.rows)
        if len(response.rows) < 100000:
            break
        monthly_request.offset = len(all_rows)

    # Process monthly data
    monthly_data = {}
    total_monthly_views = {}

    for row in all_rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        page_path = row.dimension_values[2].value
        views = int(row.metric_values[0].value)

        month_key = f"{month:02d}, {year}"

        if month_key not in monthly_data:
            monthly_data[month_key] = {path: 0 for path in consolidated_paths}
            total_monthly_views[month_key] = 0

        # Normalize path and add views
        normalized_path = page_path.rstrip('/')
        if normalized_path in ["/", "/home"]:
            normalized_path = "/"
            monthly_data[month_key]["/"] += views
        elif page_path in consolidated_paths:
            monthly_data[month_key][page_path] += views

        total_monthly_views[month_key] += views

    # Create DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df['Total Pageviews'] = pd.Series(total_monthly_views)
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Apply conditional formatting
    df_with_colors, color_mapping = add_color_formatting(df)  # Reuse function

    gc = gspread.authorize(creds)

    sheet_title = 'BiomarkerKB_Top20Pages'

    # Convert DataFrame to values
    values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()

    # Update sheet
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    sheet.clear()
    sheet.update('A1', values)

    # Apply color formatting in Google Sheets
    batch_update_requests = []
    for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
        if col_name in color_mapping:
            batch_update_requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [{
                            'sheetId': sheet.id,
                            'startRowIndex': 1,
                            'startColumnIndex': col_idx - 1,
                            'endColumnIndex': col_idx
                        }],
                        'gradientRule': {
                            'minpoint': {'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404}, 'type': 'MIN'},
                            'midpoint': {'color': {'red': 1, 'green': 1, 'blue': 1}, 'type': 'PERCENTILE', 'value': '50'},
                            'maxpoint': {'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420}, 'type': 'MAX'}
                        }
                    }
                }
            })

    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()

    print("Top 20 pages report created successfully for BiomarkerKB Portal with conditional formatting.")

# Execute the function
get_biomarkerkb_top_pages_overview()
# %%
def add_top_biomarkerkb_pages_chart():
    SHEET_TITLE = 'BiomarkerKB_Top20Pages'  # Updated sheet title for biomarkerkb Portal
    
    # Get sheet ID for biomarkerkb_Top20Pages
    spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    sheet_id = None
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == SHEET_TITLE:
            sheet_id = sheet['properties']['sheetId']
            break
            
    if sheet_id is None:
        raise ValueError(f"Sheet '{SHEET_TITLE}' not found")

    # Get the data to determine dimensions
    gc = gspread.authorize(creds)
    worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TITLE)
    data = worksheet.get_all_values()
    num_rows = len(data)
    num_columns = len(data[0])

    # Adjust column width for better visualization
    padding_request = {
        'requests': [{
            'updateDimensionProperties': {
                'range': {
                    'sheetId': sheet_id,
                    'dimension': 'COLUMNS',
                    'startIndex': num_columns,
                    'endIndex': num_columns + 5
                },
                'properties': {
                    'pixelSize': 75
                },
                'fields': 'pixelSize'
            }
        }]
    }

    # Define colors for the top pages
    colors = [
        {'red': 0.4, 'green': 0.4, 'blue': 1.0},  # Blue
        {'red': 1.0, 'green': 0.4, 'blue': 0.4},  # Red
        {'red': 0.4, 'green': 1.0, 'blue': 0.4},  # Green
        {'red': 1.0, 'green': 0.8, 'blue': 0.2},  # Yellow
        {'red': 0.8, 'green': 0.4, 'blue': 0.8},  # Purple
        {'red': 0.4, 'green': 0.8, 'blue': 1.0},  # Light Blue
        {'red': 1.0, 'green': 0.6, 'blue': 0.4},  # Orange
        {'red': 0.6, 'green': 0.4, 'blue': 0.2},  # Brown
        {'red': 0.8, 'green': 0.8, 'blue': 0.4},  # Light Yellow
        {'red': 0.4, 'green': 0.8, 'blue': 0.6}   # Teal
    ]

    # Create chart specification
    chart = {
        'spec': {
            'title': 'Top Pages Views Over Time - biomarkerkb Portal',  # Updated chart title
            'basicChart': {
                'chartType': 'LINE',
                'legendPosition': 'RIGHT_LEGEND',
                'headerCount': 1,
                'axis': [
                    {
                        'position': 'BOTTOM_AXIS',
                        'title': 'Month-Year'
                    },
                    {
                        'position': 'LEFT_AXIS',
                        'title': 'Page Views'
                    }
                ],
                'domains': [{
                    'domain': {
                        'sourceRange': {
                            'sources': [{
                                'sheetId': sheet_id,
                                'startRowIndex': 0,
                                'endRowIndex': num_rows,
                                'startColumnIndex': 0,
                                'endColumnIndex': 1
                            }]
                        }
                    },
                    'reversed': True
                }],
                'series': []
            }
        },
        'position': {
            'overlayPosition': {
                'anchorCell': {
                    'sheetId': sheet_id,
                    'rowIndex': 0,
                    'columnIndex': num_columns + 3
                },
                'widthPixels': 900,
                'heightPixels': 520
            }
        }
    }

    # Add series for the top 10 pages (columns 1 to 11, including Total Pageviews)
    for idx in range(1, 11):
        series = {
            'series': {
                'sourceRange': {
                    'sources': [{
                        'sheetId': sheet_id,
                        'startRowIndex': 0,
                        'endRowIndex': num_rows,
                        'startColumnIndex': idx,
                        'endColumnIndex': idx + 1
                    }]
                }
            },
            'targetAxis': 'LEFT_AXIS',
            'color': colors[idx - 1] if idx - 1 < len(colors) else {'red': 0.5, 'green': 0.5, 'blue': 0.5},
            'lineStyle': {'type': 'SOLID', 'width': 2}
        }
        chart['spec']['basicChart']['series'].append(series)

    # Create the chart request
    chart_request = {
        'requests': [{
            'addChart': {
                'chart': chart
            }
        }]
    }

    # Execute the request
    try:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body=chart_request
        ).execute()
        print("Chart added successfully for biomarkerkb Portal Top 20 Pages.")
    except Exception as e:
        print(f"Error creating chart: {str(e)}")

# Execute the function
add_top_biomarkerkb_pages_chart()

#%%
def create_biomarkerkb_portal_top_referrals_trend_report():
    subdomain_filter = FilterExpression(
    or_group=FilterExpressionList(
        expressions=[
            FilterExpression(
                filter=Filter(
                    field_name="hostname",
                    string_filter=Filter.StringFilter(
                        value="biomarkerkb.org",
                        match_type=Filter.StringFilter.MatchType.EXACT
                    )
                )
            ),
            FilterExpression(
                filter=Filter(
                    field_name="hostname",
                    string_filter=Filter.StringFilter(
                        value="www.biomarkerkb.org",
                        match_type=Filter.StringFilter.MatchType.EXACT
                    )
                )
            )
        ]
    )
)


    # First request: Get overall top 10 referral sources for biomarkerkb Portal
    top_referrals_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="sessionSource"),
            Dimension(name="sessionMedium")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[OrderBy(metric={"metric_name": "sessions"}, desc=True)],
        limit=10,
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        dimension_filter=FilterExpression(
            and_group=FilterExpressionList(
                expressions=[
                    FilterExpression(
                        filter=Filter(
                            field_name="sessionMedium",
                            string_filter=Filter.StringFilter(value="referral", match_type=Filter.StringFilter.MatchType.EXACT)
                        )
                    ),
                    subdomain_filter  # Apply hostname filter
                ]
            )
        )
    )

    top_referrals_response = client.run_report(top_referrals_request)
    top_referrals = [(row.dimension_values[0].value, row.dimension_values[1].value) 
                     for row in top_referrals_response.rows]

    # Second request: Get monthly data for these referral sources
    monthly_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        limit=50000,
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    monthly_response = client.run_report(monthly_request)

    # Process monthly data
    monthly_data = {}
    
    # Create all month-year combinations
    current_date = pd.Timestamp.now()
    start_date = pd.Timestamp('2023-04-01')
    date_range = pd.date_range(start=start_date, end=current_date, freq='M')
    
    # Initialize the dictionary with all possible months
    for date in date_range:
        month_key = f"{date.month:02d}, {date.year}"
        monthly_data[month_key] = {referral[0]: 0 for referral in top_referrals}

    # Fill in the actual data
    for row in monthly_response.rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        source = row.dimension_values[2].value
        sessions = int(row.metric_values[0].value)
        
        month_key = f"{month:02d}, {year}"
        if month_key in monthly_data and source in monthly_data[month_key]:
            monthly_data[month_key][source] = sessions

    # Create DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date (latest first)
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Filter out rows where all numeric columns are 0
    numeric_columns = df.columns.drop('Month-Year')
    df = df[~(df[numeric_columns] == 0).all(axis=1)]

    # Apply conditional formatting
    df_with_colors, color_mapping = add_color_formatting(df)  # Reuse the existing function

    sheet_title = 'BiomarkerKB_Top10Referrals'

    # Convert DataFrame to values for Google Sheets
    values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()

    # Update the sheet
    request = service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f'{sheet_title}!A1',
        valueInputOption='RAW',
        body={'values': values}
    )
    response = request.execute()

    # Apply color formatting in Google Sheets
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)

    batch_update_requests = []
    for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
        if col_name in color_mapping:
            batch_update_requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [{
                            'sheetId': sheet.id,
                            'startRowIndex': 1,  # Skip header row
                            'startColumnIndex': col_idx - 1,
                            'endColumnIndex': col_idx
                        }],
                        'gradientRule': {
                            'minpoint': {
                                'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                                'type': 'MIN'
                            },
                            'midpoint': {
                                'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                                'type': 'PERCENTILE',
                                'value': '50'
                            },
                            'maxpoint': {
                                'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                                'type': 'MAX'
                            }
                        }
                    }
                }
            })

    # Execute batch update for conditional formatting
    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()

    print("Top referrals trend report created successfully for biomarkerkb Portal with conditional formatting.")

# Execute the function
create_biomarkerkb_portal_top_referrals_trend_report()
# %%
def create_biomarkerkb_portal_top_countries_report():
    try:
        # Define filter for biomarkerkb Portal (Exact match for www.biomarkerkb.org and biomarkerkb.org)
        subdomain_filter = FilterExpression(
            or_group=FilterExpressionList(
                expressions=[
                    FilterExpression(
                        filter=Filter(
                            field_name="hostname",
                            string_filter=Filter.StringFilter(
                                value="biomarkerkb.org",
                                match_type=Filter.StringFilter.MatchType.EXACT
                            )
                        )
                    ),
                    FilterExpression(
                        filter=Filter(
                            field_name="hostname",
                            string_filter=Filter.StringFilter(
                                value="www.biomarkerkb.org",
                                match_type=Filter.StringFilter.MatchType.EXACT
                            )
                        )
                    )
                ]
            )
        )

        # First request: Get top 10 countries overall (filtered for biomarkerkb)
        top_countries_request = RunReportRequest(
            property=f'properties/{property_id}',
            dimensions=[Dimension(name="country")],
            metrics=[Metric(name="engagedSessions")],
            order_bys=[OrderBy(metric={"metric_name": "engagedSessions"}, desc=True)],
            limit=10,
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
            dimension_filter=subdomain_filter  # Apply hostname filter
        )

        top_countries_response = client.run_report(top_countries_request)
        top_countries = [row.dimension_values[0].value for row in top_countries_response.rows]

        # Second request: Get monthly data for these top countries
        monthly_request = RunReportRequest(
            property=f'properties/{property_id}',
            dimensions=[
                Dimension(name="year"),
                Dimension(name="month"),
                Dimension(name="country")
            ],
            metrics=[Metric(name="engagedSessions")],
            order_bys=[
                OrderBy(dimension={"dimension_name": "year"}, desc=True),
                OrderBy(dimension={"dimension_name": "month"}, desc=True)
            ],
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
            dimension_filter=subdomain_filter  # Apply hostname filter
        )

        monthly_response = client.run_report(monthly_request)

        # Process monthly data
        monthly_data = {}
        total_monthly_sessions = {}

        for row in monthly_response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            country = row.dimension_values[2].value

            if country in top_countries:
                month_key = f"{month:02d}, {year}"
                if month_key not in monthly_data:
                    monthly_data[month_key] = {country: 0 for country in top_countries}
                    total_monthly_sessions[month_key] = 0

                sessions = int(row.metric_values[0].value)
                monthly_data[month_key][country] = sessions
                total_monthly_sessions[month_key] += sessions

        # Create DataFrame
        df = pd.DataFrame.from_dict(monthly_data, orient='index')
        df['Total Engaged Sessions'] = pd.Series(total_monthly_sessions)
        df.index.name = 'Month-Year'
        df = df.reset_index()

        # Sort by date
        df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
        df = df.sort_values('Sort_Date', ascending=False)
        df = df.drop('Sort_Date', axis=1)

        # Reorder columns to put Total first
        cols = ['Month-Year', 'Total Engaged Sessions'] + [col for col in df.columns if col not in ['Month-Year', 'Total Engaged Sessions']]
        df = df[cols]

        # Apply conditional formatting
        df_with_colors, color_mapping = add_color_formatting(df)  # Reuse function

        gc = gspread.authorize(creds)

        sheet_title = 'biomarkerkb_Top10Countries_Monthly'  # Updated for biomarkerkb

        # Create or get worksheet
        try:
            worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
        except:
            worksheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(sheet_title, rows=100, cols=20)

        # Clear existing content
        worksheet.clear()

        # Update values
        values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()
        worksheet.update('A1', values, value_input_option='RAW')

        # Apply color formatting in Google Sheets
        batch_update_requests = []
        for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
            if col_name in color_mapping:
                batch_update_requests.append({
                    'addConditionalFormatRule': {
                        'rule': {
                            'ranges': [{
                                'sheetId': worksheet.id,
                                'startRowIndex': 1,
                                'startColumnIndex': col_idx - 1,
                                'endColumnIndex': col_idx
                            }],
                            'gradientRule': {
                                'minpoint': {
                                    'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                                    'type': 'MIN'
                                },
                                'midpoint': {
                                    'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                                    'type': 'PERCENTILE',
                                    'value': '50'
                                },
                                'maxpoint': {
                                    'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                                    'type': 'MAX'
                                }
                            }
                        }
                    }
                })

        # Execute batch update for conditional formatting
        if batch_update_requests:
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={'requests': batch_update_requests}
            ).execute()

        print("Top 10 countries report created successfully for biomarkerkb Portal with conditional formatting.")

    except Exception as e:
        print(f"Error creating report: {str(e)}")

# Execute the function
create_biomarkerkb_portal_top_countries_report()
# %%
##BiomarkerKB Data
def create_biomarkerkbdata_ga4_report():
    # Main GA4 metrics request
    main_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[Dimension(name="year"), Dimension(name="month")],
        metrics=[
            Metric(name="totalUsers"),
            Metric(name="activeUsers"),
            Metric(name="newUsers"),
            Metric(name="eventCount"),
            Metric(name="sessions")
        ],
        order_bys=[OrderBy(dimension={'dimension_name': 'month'})],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        dimension_filter={
            'filter': {
                'field_name': 'hostname',
                'string_filter': {
                    'value': 'data.biomarkerkb.org',
                    'match_type': 'EXACT'
                }
            }
        }
    )

    # Traffic sources request
    traffic_source_request = RunReportRequest(
        property='properties/' + property_id,
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[
            OrderBy(dimension={'dimension_name': 'year'}, desc=True),
            OrderBy(dimension={'dimension_name': 'month'}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2020-01-01", end_date="today")],
        dimension_filter={
            'filter': {
                'field_name': 'hostname',
                'string_filter': {
                    'value': 'data.biomarkerkb.org',
                    'match_type': 'EXACT'
                }
            }
        }
    )

    # Send requests
    main_response = client.run_report(main_request)
    traffic_source_response = client.run_report(traffic_source_request)

    # Process main metrics
    def process_biomarkerkbdata_metrics(response):
        row_headers = [row.dimension_values for row in response.rows]
        metric_values = [row.metric_values for row in response.rows]

        data = []
        
        for i in range(len(row_headers)):
            year = int(row_headers[i][0].value)
            month = int(row_headers[i][1].value)
            total_users = float(metric_values[i][0].value)
            active_users = float(metric_values[i][1].value)
            new_users = float(metric_values[i][2].value)
            returning_users = total_users - new_users
            hits_events = float(metric_values[i][3].value)
            sessions = float(metric_values[i][4].value)

            data.append([year, month, total_users, active_users, returning_users, new_users, hits_events, sessions])

        df = pd.DataFrame(data, columns=[
            "Year", "Month", "Total Users", "Users/Active Users", "Returning Users", "New Users", "Hits/Events", "Sessions"
        ])

        return df

    # Process traffic sources
    def process_biomarkerkbdata_traffic_sources(response):
        data = {}
        for row in response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            source = row.dimension_values[2].value
            sessions = float(row.metric_values[0].value)
            
            key = f"{month:02}, {year}"
            if key not in data:
                data[key] = {"Organic Search": 0, "Direct": 0, "Referral": 0}
            
            if source.lower() == "google":
                data[key]["Organic Search"] += sessions
            elif source.lower() == "(direct)":
                data[key]["Direct"] += sessions
            elif source.lower() not in ["google", "(direct)"]:
                data[key]["Referral"] += sessions

        df = pd.DataFrame.from_dict(data, orient='index', columns=["Organic Search", "Direct", "Referral"])
        df.index.name = "Month-Year"
        df = df.reset_index()
        
        return df

    # Process both datasets
    main_df = process_biomarkerkbdata_metrics(main_response)
    traffic_sources_df = process_biomarkerkbdata_traffic_sources(traffic_source_response)

    # Combine datasets
    def combine_datasets(main_df, traffic_sources_df):
        # Create Month-Year column for both dataframes
        main_df['Month-Year'] = main_df['Month'].apply(lambda x: f'{x:02}') + ', ' + main_df['Year'].astype(str)
        
        # Merge dataframes
        combined_df = pd.merge(main_df, traffic_sources_df, on='Month-Year', how='left')
        
        # Reorder and select columns
        columns_order = [
            'Month-Year', 'Total Users', 'Users/Active Users', 'Returning Users', 
            'New Users', 'Hits/Events', 'Sessions', 
            'Organic Search', 'Direct', 'Referral'
        ]
        combined_df = combined_df[columns_order]
        
        # Create a datetime column for sorting
        combined_df['Sort_Date'] = pd.to_datetime(combined_df['Month-Year'], format='%m, %Y')
        
        # Sort in descending order (latest first)
        combined_df = combined_df.sort_values('Sort_Date', ascending=False)
        
        # Drop the sorting column
        combined_df = combined_df.drop(columns=['Sort_Date'])
        
        return combined_df

    return combine_datasets(main_df, traffic_sources_df)

def add_color_formatting(df):
    """
    Add color formatting based on trends and outliers
    Color coding:
    - Green: Above average (positive trend)
    - Red: Below average (negative trend)
    - Yellow: Slightly different from average
    """
    def get_color_class(column):
        # Calculate mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()
        
        def color_mapper(value):
            # More than 1 std dev above mean
            if value > mean + std:
                return 'positive-high-outlier'
            # Between 0.5 and 1 std dev above mean
            elif value > mean + (std/2):
                return 'positive-mild-outlier'
            # More than 1 std dev below mean
            elif value < mean - std:
                return 'negative-high-outlier'
            # Between 0.5 and 1 std dev below mean
            elif value < mean - (std/2):
                return 'negative-mild-outlier'
            # Close to average
            else:
                return 'average'
        
        return df[column].apply(color_mapper)
    
    # Columns to analyze (excluding Month-Year)
    numeric_columns = df.columns.drop('Month-Year').tolist()
    
    # Create color mapping for each column
    color_mapping = {col: get_color_class(col) for col in numeric_columns}
    
    return df, color_mapping

# Google Sheets API setup and export
def export_to_google_sheets(df, color_mapping):
    gc = gspread.authorize(creds)
    SHEET_TITLE = 'biomarkerkb_Data_Overview'

    # Check if sheet exists, if not create it
    try:
        sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_TITLE)
    except gspread.exceptions.WorksheetNotFound:
        sheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(title=SHEET_TITLE, rows="100", cols="20")

    # Convert DataFrame to list of lists for Google Sheets
    values = [df.columns.tolist()] + df.values.tolist()

    # Update the sheet
    sheet.clear()  # Clear existing content
    sheet.update('A1', values, value_input_option='RAW')

    # Formatting colors
    batch_update_requests = [{
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': sheet.id,
                        'startRowIndex': 1,  # Skip header row
                        'startColumnIndex': col_idx - 1,
                        'endColumnIndex': col_idx
                    }],
                    'gradientRule': {
                        'minpoint': {
                            'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                            'type': 'MIN'
                        },
                        'midpoint': {
                            'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                            'type': 'PERCENTILE',
                            'value': '50'
                        },
                        'maxpoint': {
                            'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                            'type': 'MAX'
                        }
                    }
                }
            }
        } for col_idx, col_name in enumerate(df.columns[1:], start=2)]  # Skip first column

    # Execute batch update
    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()

    print(f"Report updated successfully in sheet: {SHEET_TITLE}")

# Main execution
df = create_biomarkerkbdata_ga4_report()
df_with_colors, color_mapping = add_color_formatting(df)
export_to_google_sheets(df_with_colors, color_mapping)

# Optional: Print the first few rows and color mapping
print(df_with_colors.head())
print("\nColor Mapping Legend:")
print("- Green shades: Performance above average (light to dark intensity)")
print("- Red shades: Performance below average (light to dark intensity)")
print("- White: Performance close to average")
# %%
def get_biomarkerkb_data_top_pages_overview():

    # Define filter for BiomarkerKB Portal (Exact match for www.biomarkerkb.org and biomarkerkb.org)
    subdomain_filter = FilterExpression(
        filter=Filter(
            field_name="hostname",
            string_filter=Filter.StringFilter(
                value="data.biomarkerkb.org",
                match_type=Filter.StringFilter.MatchType.EXACT
            )
        )
    )

    # First request: Get top 20 pages overall with extended date range (filtered for BiomarkerKB)
    top_pages_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="pagePath"),
            Dimension(name="pageTitle")
        ],
        metrics=[Metric(name="screenPageViews")],
        order_bys=[OrderBy(metric={"metric_name": "screenPageViews"}, desc=True)],
        limit=100000,
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    top_pages_response = client.run_report(top_pages_request)

    # Process path mapping
    path_mapping = {}
    for row in top_pages_response.rows:
        path = row.dimension_values[0].value
        normalized_path = path.rstrip('/')

        # Handle special cases and duplicates
        if path in ["/", "/home", "/home/"]:
            normalized_path = "/"

        # Add views to get proper top pages
        views = int(row.metric_values[0].value)
        if normalized_path not in path_mapping:
            path_mapping[normalized_path] = {'path': path, 'views': views}
        else:
            path_mapping[normalized_path]['views'] += views

    # Sort by total views and get top 20
    consolidated_paths = [info['path'] for info in sorted(path_mapping.values(), key=lambda x: x['views'], reverse=True)][:20]

    # Second request: Get monthly data for these top pages
    monthly_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="pagePath")
        ],
        metrics=[Metric(name="screenPageViews")],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        limit=100000,
        offset=0,
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    # Get all data by handling pagination
    all_rows = []
    while True:
        response = client.run_report(monthly_request)
        all_rows.extend(response.rows)
        if len(response.rows) < 100000:
            break
        monthly_request.offset = len(all_rows)

    # Process monthly data
    monthly_data = {}
    total_monthly_views = {}

    for row in all_rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        page_path = row.dimension_values[2].value
        views = int(row.metric_values[0].value)

        month_key = f"{month:02d}, {year}"

        if month_key not in monthly_data:
            monthly_data[month_key] = {path: 0 for path in consolidated_paths}
            total_monthly_views[month_key] = 0

        # Normalize path and add views
        normalized_path = page_path.rstrip('/')
        if normalized_path in ["/", "/home"]:
            normalized_path = "/"
            monthly_data[month_key]["/"] += views
        elif page_path in consolidated_paths:
            monthly_data[month_key][page_path] += views

        total_monthly_views[month_key] += views

    # Create DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df['Total Pageviews'] = pd.Series(total_monthly_views)
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Apply conditional formatting
    df_with_colors, color_mapping = add_color_formatting(df)  # Reuse function

    gc = gspread.authorize(creds)

    sheet_title = 'BiomarkerKB_Data_Top20Pages'

    # Convert DataFrame to values
    values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()

    # Update sheet
    sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    sheet.clear()
    sheet.update('A1', values)

    # Apply color formatting in Google Sheets
    batch_update_requests = []
    for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
        if col_name in color_mapping:
            batch_update_requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [{
                            'sheetId': sheet.id,
                            'startRowIndex': 1,
                            'startColumnIndex': col_idx - 1,
                            'endColumnIndex': col_idx
                        }],
                        'gradientRule': {
                            'minpoint': {'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404}, 'type': 'MIN'},
                            'midpoint': {'color': {'red': 1, 'green': 1, 'blue': 1}, 'type': 'PERCENTILE', 'value': '50'},
                            'maxpoint': {'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420}, 'type': 'MAX'}
                        }
                    }
                }
            })

    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()

    print("Top 20 pages report created successfully for BiomarkerKB Data with conditional formatting.")

# Execute the function
get_biomarkerkb_data_top_pages_overview()
# %%
## Top 10 Referrals - biomarkerkb Data
def create_biomarkerkbdata_top_referrals_trend_report():
    client = BetaAnalyticsDataClient()
    property_id = "361964108"
    subdomain_filter = FilterExpression(
        filter=Filter(
            field_name="hostname",
            string_filter=Filter.StringFilter(
                value="data.biomarkerkb.org",
                match_type=Filter.StringFilter.MatchType.EXACT
            )
        )
    )

    # First request: Get overall top 10 referral sources for biomarkerkb Data
    top_referrals_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="sessionSource"),
            Dimension(name="sessionMedium")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[OrderBy(metric={"metric_name": "sessions"}, desc=True)],
        limit=10,
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        dimension_filter=FilterExpression(
            and_group=FilterExpressionList(
                expressions=[
                    FilterExpression(
                        filter=Filter(
                            field_name="sessionMedium",
                            string_filter=Filter.StringFilter(value="referral", match_type=Filter.StringFilter.MatchType.EXACT)
                        )
                    ),
                    subdomain_filter  # Apply hostname filter
                ]
            )
        )
    )

    top_referrals_response = client.run_report(top_referrals_request)
    top_referrals = [(row.dimension_values[0].value, row.dimension_values[1].value) 
                     for row in top_referrals_response.rows]

    # Second request: Get monthly data for these referral sources
    monthly_request = RunReportRequest(
        property=f'properties/{property_id}',
        dimensions=[
            Dimension(name="year"),
            Dimension(name="month"),
            Dimension(name="sessionSource")
        ],
        metrics=[Metric(name="sessions")],
        order_bys=[
            OrderBy(dimension={"dimension_name": "year"}, desc=True),
            OrderBy(dimension={"dimension_name": "month"}, desc=True)
        ],
        date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
        limit=50000,
        dimension_filter=subdomain_filter  # Apply hostname filter
    )

    monthly_response = client.run_report(monthly_request)

    # Process monthly data
    monthly_data = {}
    
    # Create all month-year combinations
    current_date = pd.Timestamp.now()
    start_date = pd.Timestamp('2023-04-01')
    date_range = pd.date_range(start=start_date, end=current_date, freq='M')
    
    # Initialize the dictionary with all possible months
    for date in date_range:
        month_key = f"{date.month:02d}, {date.year}"
        monthly_data[month_key] = {referral[0]: 0 for referral in top_referrals}

    # Fill in the actual data
    for row in monthly_response.rows:
        year = int(row.dimension_values[0].value)
        month = int(row.dimension_values[1].value)
        source = row.dimension_values[2].value
        sessions = int(row.metric_values[0].value)
        
        month_key = f"{month:02d}, {year}"
        if month_key in monthly_data and source in monthly_data[month_key]:
            monthly_data[month_key][source] = sessions

    # Create DataFrame
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    df.index.name = 'Month-Year'
    df = df.reset_index()

    # Sort by date (latest first)
    df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
    df = df.sort_values('Sort_Date', ascending=False)
    df = df.drop('Sort_Date', axis=1)

    # Filter out rows where all numeric columns are 0
    numeric_columns = df.columns.drop('Month-Year')
    df = df[~(df[numeric_columns] == 0).all(axis=1)]

    # Apply conditional formatting
    df_with_colors, color_mapping = add_color_formatting(df)  # Reuse the existing function

    sheet_title = 'biomarkerkb_Data_Top10Referrals'

    # Check if sheet exists, if not, create it
    gc = gspread.authorize(creds)
    try:
        sheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
    except gspread.exceptions.WorksheetNotFound:
        sheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(title=sheet_title, rows="100", cols="20")

    # Convert DataFrame to values for Google Sheets
    values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()

    # Update the sheet
    sheet.clear()
    sheet.update('A1', values)

    # Apply color formatting in Google Sheets
    batch_update_requests = []
    for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
        if col_name in color_mapping:
            batch_update_requests.append({
                'addConditionalFormatRule': {
                    'rule': {
                        'ranges': [{
                            'sheetId': sheet.id,
                            'startRowIndex': 1,  # Skip header row
                            'startColumnIndex': col_idx - 1,
                            'endColumnIndex': col_idx
                        }],
                        'gradientRule': {
                            'minpoint': {
                                'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                                'type': 'MIN'
                            },
                            'midpoint': {
                                'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                                'type': 'PERCENTILE',
                                'value': '50'
                            },
                            'maxpoint': {
                                'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                                'type': 'MAX'
                            }
                        }
                    }
                }
            })

    # Execute batch update for conditional formatting
    if batch_update_requests:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={'requests': batch_update_requests}
        ).execute()

    print("Top referrals trend report created successfully for biomarkerkb Data with conditional formatting.")

# Execute the function
create_biomarkerkbdata_top_referrals_trend_report()
#%%
## Top 10 Countries - biomarkerkb Data
def create_biomarkerkbdata_top_countries_report():
    try:
        client = BetaAnalyticsDataClient()
        property_id = "361964108"

        # Define filter for biomarkerkb Data (Exact match for data.biomarkerkb.org)
        subdomain_filter = FilterExpression(
            filter=Filter(
                field_name="hostname",
                string_filter=Filter.StringFilter(
                    value="data.biomarkerkb.org",
                    match_type=Filter.StringFilter.MatchType.EXACT
                )
            )
        )

        # First request: Get top 10 countries overall (filtered for biomarkerkb Data)
        top_countries_request = RunReportRequest(
            property=f'properties/{property_id}',
            dimensions=[Dimension(name="country")],
            metrics=[Metric(name="engagedSessions")],
            order_bys=[OrderBy(metric={"metric_name": "engagedSessions"}, desc=True)],
            limit=10,
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
            dimension_filter=subdomain_filter  # Apply hostname filter
        )

        top_countries_response = client.run_report(top_countries_request)
        top_countries = [row.dimension_values[0].value for row in top_countries_response.rows]

        # Second request: Get monthly data for these top countries
        monthly_request = RunReportRequest(
            property=f'properties/{property_id}',
            dimensions=[
                Dimension(name="year"),
                Dimension(name="month"),
                Dimension(name="country")
            ],
            metrics=[Metric(name="engagedSessions")],
            order_bys=[
                OrderBy(dimension={"dimension_name": "year"}, desc=True),
                OrderBy(dimension={"dimension_name": "month"}, desc=True)
            ],
            date_ranges=[DateRange(start_date="2023-04-01", end_date="today")],
            dimension_filter=subdomain_filter  # Apply hostname filter
        )

        monthly_response = client.run_report(monthly_request)

        # Process monthly data
        monthly_data = {}
        total_monthly_sessions = {}

        for row in monthly_response.rows:
            year = int(row.dimension_values[0].value)
            month = int(row.dimension_values[1].value)
            country = row.dimension_values[2].value

            if country in top_countries:
                month_key = f"{month:02d}, {year}"
                if month_key not in monthly_data:
                    monthly_data[month_key] = {country: 0 for country in top_countries}
                    total_monthly_sessions[month_key] = 0

                sessions = int(row.metric_values[0].value)
                monthly_data[month_key][country] = sessions
                total_monthly_sessions[month_key] += sessions

        # Create DataFrame
        df = pd.DataFrame.from_dict(monthly_data, orient='index')
        df['Total Engaged Sessions'] = pd.Series(total_monthly_sessions)
        df.index.name = 'Month-Year'
        df = df.reset_index()

        # Sort by date
        df['Sort_Date'] = pd.to_datetime(df['Month-Year'], format='%m, %Y')
        df = df.sort_values('Sort_Date', ascending=False)
        df = df.drop('Sort_Date', axis=1)

        # Reorder columns to put Total first
        cols = ['Month-Year', 'Total Engaged Sessions'] + [col for col in df.columns if col not in ['Month-Year', 'Total Engaged Sessions']]
        df = df[cols]

        # Apply conditional formatting
        df_with_colors, color_mapping = add_color_formatting(df)  # Reuse function

        gc = gspread.authorize(creds)

        SPREADSHEET_ID = '1faXFb6yEYzHssBFU-LH5b4YIRhBxttfBuyaHnl09fuA'
        sheet_title = 'biomarkerkb_Data_Top10Countries_Monthly'  # Updated for biomarkerkb Data

        # Create or get worksheet
        try:
            worksheet = gc.open_by_key(SPREADSHEET_ID).worksheet(sheet_title)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = gc.open_by_key(SPREADSHEET_ID).add_worksheet(title=sheet_title, rows="100", cols="20")

        # Clear existing content
        worksheet.clear()

        # Update values
        values = [df_with_colors.columns.tolist()] + df_with_colors.values.tolist()
        worksheet.update('A1', values, value_input_option='RAW')

        # Apply color formatting in Google Sheets
        batch_update_requests = []
        for col_idx, col_name in enumerate(df_with_colors.columns[1:], start=2):  # Skip first column
            if col_name in color_mapping:
                batch_update_requests.append({
                    'addConditionalFormatRule': {
                        'rule': {
                            'ranges': [{
                                'sheetId': worksheet.id,
                                'startRowIndex': 1,  # Skip header row
                                'startColumnIndex': col_idx - 1,
                                'endColumnIndex': col_idx
                            }],
                            'gradientRule': {
                                'minpoint': {
                                    'color': {'red': 0.839, 'green': 0.404, 'blue': 0.404},  # Red
                                    'type': 'MIN'
                                },
                                'midpoint': {
                                    'color': {'red': 1, 'green': 1, 'blue': 1},  # White
                                    'type': 'PERCENTILE',
                                    'value': '50'
                                },
                                'maxpoint': {
                                    'color': {'red': 0.420, 'green': 0.655, 'blue': 0.420},  # Green
                                    'type': 'MAX'
                                }
                            }
                        }
                    }
                })

        # Execute batch update for conditional formatting
        if batch_update_requests:
            service.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={'requests': batch_update_requests}
            ).execute()

        print("Top 10 countries report created successfully for biomarkerkb Data with conditional formatting.")

    except Exception as e:
        print(f"Error creating report: {str(e)}")

# Execute the function
create_biomarkerkbdata_top_countries_report()
# %%
