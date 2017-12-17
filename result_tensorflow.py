"""
Create the true answer csv for the test data
The answer 1 correspond to malignant and 1 to benign
"""
import os, re
import pandas as pd


def create_test_list_indices():
    """
    Create indices' list for test pictures
    :return: List of indicices [(str), (str), ...]
    """
    indices_list = []
    for image_indice in os.listdir('test'):
        indices_list.append(re.sub(r'\.jpg$', '', image_indice))
    return indices_list


def create_result_for_test(test_indices_list):
    """
    Creation of the result csv for the test data
    :param test_indices_list: path of the metadata csv from ISIC (complete)
    """
    # Define the lists of result
    list_name = []
    list_result = []
    list_binary_result = []

    # For all the pictures of the metadata csv file take name (ID) and answer (benign/malignant)
    for name, result in zip(data['name'], data['benign_malignant']):
        # Regex the name
        name = re.sub(r'\.jpg$', '', name)
        name = re.sub(r'^ISIC_', '', name)

        # If this picture belong to the test add it and its result
        if name in test_indices_list:
            if result == 'benign':
                list_name.append(name)
                list_result.append(result)
                list_binary_result.append("1")
            else:
                list_name.append(name)
                list_result.append(result)
                list_binary_result.append("0")

    # Create the binary answer dataframe
    result_binary = pd.concat([pd.Series(list_name), pd.Series(list_result), pd.Series(list_binary_result)], axis=1)

    # Save them as csv
    pd.DataFrame(result_binary).to_csv('result_test.csv', sep=',', index=False)


if __name__ == "__main__":
    # If the answer csv exist
    if os.path.isfile('result_test.csv'):
        print("\nThe results already exists, delete 'result_test_save.csv' file in  order to recreate them")
    else:
        # Load the metadat csv
        data = pd.read_csv('metadata.csv', sep=",")
        # Create the test indices list
        test_indices_list = create_test_list_indices()
        # Create the result answer (from ISIC)
        create_result_for_test(test_indices_list)