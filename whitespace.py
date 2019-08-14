
def remove_whitespace(df):
    for row in df.loc[:"Requirements"]:
        df = df.replace('\n','',regex=True)
    return df

def remove_duplicates(df):
    df.drop_duplicates(subset = "Requirements", keep = 'first', inplace = True)
    return df