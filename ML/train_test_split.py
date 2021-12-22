from sklearn.model_selection import train_test_split

SUBSTRATE_TYPES = ['ag_rated', 'au_rated']


def splitting_data(data_for_ml):
    """
    Train test split method used on features and labels
    :param data_for_ml: dict
    :return: dict
    """
    
    split_data = {}
    for substrate_type in SUBSTRATE_TYPES:
        data_for_ml[substrate_type].round()
        X_train, X_test, y_train, y_test = train_test_split(data_for_ml[substrate_type].iloc[:, 1:-4],
                                                            data_for_ml[substrate_type].loc[:, 'Quality'],
                                                            test_size=0.2, random_state=0)
        
        split_data[substrate_type] = X_train, X_test, y_train, y_test
    return split_data
