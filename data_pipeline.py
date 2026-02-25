import pandas as pd
transaction_df = pd.read_excel('Tourism Dataset/Transaction.xlsx')
user_df = pd.read_excel('Tourism Dataset/User.xlsx')
city_df = pd.read_excel('Tourism Dataset/City.xlsx')
type_df = pd.read_excel('Tourism Dataset/Type.xlsx')  
visit_mode_df = pd.read_excel('Tourism Dataset/Mode.xlsx')
continent_df = pd.read_excel('Tourism Dataset/Continent.xlsx')
country_df = pd.read_excel('Tourism Dataset/Country.xlsx')
region_df = pd.read_excel('Tourism Dataset/Region.xlsx')
item_df = pd.read_excel('Tourism Dataset/Item.xlsx')

print("City Columns:", city_df.columns.tolist())
print("Country Columns:", country_df.columns.tolist())

# 2. Pre-Cleaning: Strip spaces from all headers 
for df in [transaction_df, user_df, city_df, type_df, visit_mode_df, 
           continent_df, country_df, region_df, item_df]:
    df.columns = df.columns.str.strip()

# 3. Step-by-Step Merging 
# Build Geography
geo = city_df.merge(country_df, on='CountryId', how='left') \
             .merge(region_df, on='RegionId', how='left') \
             .merge(continent_df, on='ContinentId', how='left')

# Build User Master
user_master = user_df.merge(geo, on='CityId', how='left')

# Build Item Master
item_master = item_df.merge(type_df, on='AttractionTypeId', how='left')
# Convert the merge keys to strings to fix the data type mismatch
transaction_df['VisitMode'] = transaction_df['VisitMode'].astype(str).str.strip()
visit_mode_df['VisitMode'] = visit_mode_df['VisitMode'].astype(str).str.strip()

# Now perform the merge
final_df = transaction_df.merge(user_master, on='UserId', how='left') \
                         .merge(item_master, on='AttractionId', how='left') \
                         .merge(visit_mode_df, on='VisitMode', how='left', suffixes=('', '_drop'))

# Final Merge (Check if your Transaction table uses 'VisitMode' or 'VisitModeId')
# We use 'VisitMode' here as it's common in this dataset 
final_df = transaction_df.merge(user_master, on='UserId', how='left') \
                         .merge(item_master, on='AttractionId', how='left')

# Safe join for Visit Mode
if 'VisitModeId' in final_df.columns and 'VisitModeId' in visit_mode_df.columns:
    final_df = final_df.merge(visit_mode_df, on='VisitModeId', how='left')
else:
    # If IDs are missing, merge on the text name 
    final_df = final_df.merge(visit_mode_df, on='VisitMode', how='left', suffixes=('', '_drop'))
    final_df = final_df.loc[:, ~final_df.columns.str.contains('_drop')]

# 4. Mandatory Cleaning 
final_df['Rating'] = final_df['Rating'].fillna(final_df['Rating'].mean()) # Handle missing ratings
final_df['VisitYear'] = final_df['VisitYear'].astype(int) # Standardize Year
final_df['VisitMonth'] = final_df['VisitMonth'].astype(int) # Standardize Month

# 5. Save 
final_df.to_csv('cleaned_tourism_data.csv', index=False)
print("Phase 1 Complete! 'cleaned_tourism_data.csv' created.")