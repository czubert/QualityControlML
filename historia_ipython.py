def pokazuj(nazwa, title):
    mask = rated_spectra['ag'].index.str.contains(nazwa + '_')
    widma = rated_spectra['ag'][mask]
    px.line(widma.T).update_layout(title=title).update_yaxes(range=[-2e3, 6e4]).show('browser')
    
def rysuj(nazwa, num=''):
    mask = rated_spectra['ag'].index.str.contains(nazwa + '_')
    widma = rated_spectra['ag'][mask].iloc[:, :-3]
    return px.line(widma.T).update_layout(title=f'peak1 {num}').update_yaxes(range=[-2e3, 6e4])

    
chujowe = set(best.reset_index().sort_values('peak1')['id'].iloc[:90])
spoko = set(best.reset_index().sort_values('peak1')['id'].iloc[130:])
with open('../best_0.txt', 'w') as infile:
    for i in chujowe: 
        infile.write(i)
        infile.write('\n')

with open('../best_1.txt', 'w') as infile:
    for i in spoko: 
        infile.write(i)
        infile.write('\n')

wzystkie = set(*spoko, *chujowe)
df = rated_spectra['ag_bg']
mask = df['id'].str[:-2].isin(wzystkie)
df_wybrane['y'] = df['id'].str[:-2].isin(spoko).astype(int)
import joblib
joblib.dump(df_wybrane, '/Users/charzewski/PycharmProjects/QualityControlML/data_output/step_3_rate_data/rated_data.joblib')