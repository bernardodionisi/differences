# import pandas as pd
# import numpy as np
#
# # data = pd.read_csv("/Users/bernardodionisi/Dropbox/my-Projects/packages/differences/differences/datasets/data/mpdta.csv")
#
# data["random_weights_varying"] = np.random.uniform(low=0.0001, high=1, size=len(data))
#
# data = data.assign(
#     random_weights_by_entity=lambda x: x.groupby("countyreal")["random_weights_varying"].transform(
#         "mean"))
#
# data["first_treat"] = np.where(
#     data["first_treat"] == 0, np.nan, data["first_treat"]
# )
#


# # data.to_csv("/Users/bernardodionisi/Dropbox/my-Projects/packages/differences/differences/datasets/data/mpdta.csv", index=False)
#
#
# # data = pd.read_csv("/Users/bernardodionisi/Dropbox/my-Projects/packages/differences/differences/datasets/data/rcs-multi_period.csv")
#
# data["random_weights"] = np.random.uniform(low=0.0001, high=1, size=len(data))
#
# data["G"] = np.where(data["G"] == 0, np.nan, data["G"])
#
# # data.to_csv("/Users/bernardodionisi/Dropbox/my-Projects/packages/differences/differences/datasets/data/rcs-multi_period.csv", index=False)
