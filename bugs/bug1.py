import csv
import pandas
from anonympy.pandas import dfAnonymizer


def main():
    with open("./examples/sample_data.csv") as file:
        csvreader = csv.reader(file, delimiter=";")
        df = pandas.DataFrame(csvreader)
        anonym = dfAnonymizer(df)
        anonym.anonymize()
        print(anonym)


if __name__ == "__main__":
    main()
