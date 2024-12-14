# %%
import matplotlib.pyplot as plt
from bolero.tl.motif.jaspar import JASPARMotifDatabase
# %%
JASPARMotifDatabase.available_databases()
# %%
motif_db = JASPARMotifDatabase("CisBP_Mouse_FigR", max_length=24)
motif_db
# %%
motif = motif_db["Fos"]
motif.plot()
motif
# %%
motif.pwm.T
# %%
one_hot = motif.sample_dna_one_hot(num_sequences=10)
one_hot.shape, one_hot.dtype
# %%
motif.sample_dna_sequence(num_sequences=5)
# %%
