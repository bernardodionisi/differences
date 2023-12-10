
# def to_att_gt_pandas(
#         self,
#         base_period: Literal["varying", "universal"] = "varying",  # or 'universal'
#         anticipation: int = 0,
# ):
#     """
#     Group
#     -----
#         Export
#
#     Parameters
#     ----------
#     base_period
#     anticipation
#
#     Returns
#     -------
#
#     """
#     from differences.models.did.attgt import ATTgt
#
#     if len(self.event_data) != self.event_data[self.entity_column].n_unique():
#         raise ValueError("Entities must be unique in 'event_data'")
#
#     # join: 'data' w/ 'event_data'. cohort column / dosage to data (for old ATTgt)
#     data = self.data.join(self.event_data, on=self.entity_column, how="left")
#
#     data = data.to_pandas().set_index([self.entity_column, self.time_column])
#
#     return ATTgt(
#         data=data,
#         cohort_column=self.cohort_column,
#         dosage_column=self.dosage_column,
#         base_period=base_period,
#         anticipation=anticipation
#     )
