from copy import copy

from differences._OLD_NOW_PAGURO.info import Info, InfoList

info = Info(name="example")

print(info)


info_list = InfoList([info])

info2 = copy(info)

info.update(data_desc="data desc")

print(info_list)

print(info2)
