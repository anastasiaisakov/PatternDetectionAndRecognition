import spark as spark
from settings import *
import pandas as pd
from plotly.graph_objs import *
import plotly.io as pio
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
pd.options.plotting.backend = "plotly"
pio.renderers.default = "databricks"

data = Data()


data.default_states = spark.table(Tables.default_default_states).toPandas()
data.df = spark.table(Tables.e_f_sdl_maneuver_simulation)[['filename', 'label', 'timestamp', 'value', 'maneuver']]

data.df_phys = data.import_data(Labels.phys_labels)
data.df_bit = data.import_data(Labels.rel_labels)

file_names = data.df_bit.filename.unique()
pattern_list = data.extract_patterns(data.df_bit)

#PLOT RESULTS
filenames_to_plot = file_names[:8]
data.plot_filenames(filenames_to_plot)

data.check_and_save_patterns(pattern_list)
