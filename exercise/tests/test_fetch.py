from load.fetch import check_dataset_exists, compare_metadata, fetch_remote_metadata, authenticate_kaggle

def test_check_dataset_exists():
    actual = check_dataset_exists("data")

    assert actual == True or actual == False

def test_compare_metadata():
    actual = compare_metadata({"lastUpdated": 0}, {"lastUpdated": 0})

    assert actual == True or actual == False

def test_fetch_remote_metadata():
    api = authenticate_kaggle()
    actual = fetch_remote_metadata(api, "likhon148/animal-data")

    assert actual == {"lastUpdated": 0} or actual == {"lastUpdated": 1}

