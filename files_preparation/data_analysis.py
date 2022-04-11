import os
import plotly.express as px
import pandas as pd
import utils

peaks = {
    # beginning of the peak and end of the peak to estimate max and min values
    'peak1': ['671', '761'],
    'peak2': ['801', '881'],
    'peak3': ['970', '1031'],
}
IMG_DIR = 'data_output/images'


def run(grouped_data, peak='peak1', best_ratio=True):
    best, worst = prepare_data_for_visualisation(grouped_data)
    
    if best_ratio:
        ids = best.reset_index().sort_values(peak)['id'].to_list()[::2]
        name = f'Best ration of {peak}'
    else:
        ids = worst.reset_index().sort_values(peak)['id'].to_list()[::2]
        name = f'Worst ration of {peak}'
    
    # for i, id_ in enumerate(ids):
    #     pokazuj(grouped_data, id_, 'peak3 ' + str(i) + ' (best)')
    
    for i, id_ in enumerate(ids):
        file_name = str(i) + f' {name}'
        fig = rysuj(grouped_data, id_, file_name)
        
        if not os.path.isdir(f'../{IMG_DIR}'):
            os.makedirs(f'../{IMG_DIR}')
        
        fig.write_image(f'../{IMG_DIR}/{file_name}.png')
    
    print('Finished')
    return ids


def prepare_data_for_visualisation(grouped_data):
    # Getting relevant data
    ag_df = grouped_data['ag']  # Takes only ag spectra
    
    utils.change_col_names_type_to_str(ag_df)  # changes col names type from int to str, for .loc
    
    mask = ag_df['id'].str.startswith('s')  # mask to get only new spectra
    ag_df = ag_df[~mask]  # Takes only new spectra out of all ag spectra
    
    ratio_df = pd.DataFrame()  # DataFrame that will consists only of max/min ratio for each peak
    
    ratio_df['id'] = ag_df['id'].str.replace(r'_.*', '')
    
    # Getting ratio between max and mean value for each peak
    for name, values in peaks.items():
        ratio_df.loc[:, name] = ag_df.loc[:, values[0]:values[1]].max(axis=1) \
                                / ag_df.loc[:, values[0]:values[1]].min(axis=1)
    
    # TODO, czy sklejanie 2 widm na jednym podłożu ma sens? nie lepiej traktować to jako dwa różne wyniki?
    # Getting best ratio for each peak for each substrate
    best = ratio_df.groupby('id').max()
    worst = ratio_df.groupby('id').min()
    
    return best, worst


def pokazuj(data, nazwa, title):
    mask = data['ag'].index.str.contains(nazwa + '_')
    widma = data['ag'][mask]
    px.line(widma.T).update_layout(title=title).update_yaxes(range=[-2e3, 6e4]).show('browser')


def rysuj(data, nazwa, num=''):
    mask = data['ag'].index.str.contains(nazwa + '_')
    widma = data['ag'][mask].iloc[:, :-3]
    return px.line(widma.T).update_layout(title=f'peak1 {num}').update_yaxes(range=[-2e3, 6e4])


def data_analysis(grouped_files):
    run(grouped_files, best_ratio, peak)


if __name__ == "__main__":
    dir_path = 'data_output/step_2_group_data'
    file_name = 'grouped_data'
    grouped_data = utils.read_joblib(file_name, '../' + dir_path)
    ids = run(grouped_data)
