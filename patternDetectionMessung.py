import pd as pd
import spark
from patternDetectionSimulation import Data
# %matplotlib inline # ?
pd.options.plotting.backend = "plotly"
import plotly.io as pio
pio.renderers.default = "databricks"
from settings import *

from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

data = Data()

# EXISTS IN PATTERNDETECTIONSIMULATION but there are differences - ALSO WHERE R WE CALLING IT
def seq_frequency(df):
  df_freq = df[Labels.rel_labels]
  df_freq = df_freq.groupby(df_freq.columns.tolist()).size().reset_index().rename(columns={0: 'frequency'})
  df_freq['%frequency'].sum(axis=0, skipna=True)['incidence'] #This line is diff compared to the function in PATTERNDETECTIONSIMULATION
  return df_freq

data.default_states = spark.table(Tables.default_default_states).toPandas()
data.df = spark.table(Tables.e_f_sdl_maneuver_messung)[['filename','label','timestamp','value','Zeitangaben']]

data.df = data.df.filter((data.df.filename == "200207_1315_SEP117_PH_262.json.bz2") | (data.df.filename == "200207_0933_SEP117_PH_259.json.bz2"))

data.df_phys = data.import_data(Labels.phys_labels)
data.df = data.import_data(Labels.rel_labels)
data.df.set_index('timestamp', inplace=True)

pattern_list = data.extract_patterns(data.df)
file_names = data.df.filename.unique()

#PLOT RESULTS
filenames_to_plot = file_names[:2]
data.plot_filenames(filenames_to_plot, messung=True)

