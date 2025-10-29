from pathlib import Path

def train_data_path() -> Path:
    """
    Returns the location of train data file, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the train data file
    """
    for folder in (Path("."), Path(".."), Path("../..")):
        data_file = folder / "data" / "training.csv"
        if data_file.exists() and data_file.is_file():
            print("Train data file found in", data_file)
            return data_file
    raise Exception("Train data not found in expected locations.")

def test_data_path() -> Path:
    """
    Returns the location of test data file, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the test data file
    """
    for folder in (Path("."), Path(".."), Path("../..")):
        data_file = folder / "data" / "test.csv"
        if data_file.exists() and data_file.is_file():
            print("Test data file found in", data_file)
            return data_file
    raise Exception("Test data not found in expected locations.")

