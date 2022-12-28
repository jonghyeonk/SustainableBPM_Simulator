# Copyright 2022. Jonghyeon Ko
#
# All scripts in this folder are distributed as a free software. 
# We extended a few functions for fundamental concepts of process mining introduced in the complementary repository for BINet [1].
# 
# 1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
# ==============================================================================


from GreenBPM.dir import generate as generate_folders
from GreenBPM.process_modeling import ProcessMap
from GreenBPM.trace import *
from GreenBPM.simulation import EventLogGenerator

# create dirs if non-existent
generate_folders()
