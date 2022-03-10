
import sys
import pandas as pd

def readfile (path):
    # read pickle file
    return pd.read_pickle(path)

def prepareplot (df):
    # input dataframe
    # output plot object
    pass

def saveplot (plot):
    # input matplotlib plot object
    # output saved image as png
    pass

def main():
    # read sys arg as path
    path = sys.argv[1]

    df = readfile(path)

    plot = prepareplot(df)

    saveplot(plot)

if __name__ == "__main__":
    main()

