# ----------------------------------------------------------------------

from differences._OLD_NOW_PAGURO.info import InfoList
from differences._OLD_NOW_PAGURO.info import Info

info_list = InfoList()
info_list.append(description={"data desc": "example data"})
info_list.add_inplace(description={"data desc": "example data"})

info = Info(name="example")
print(hex(id(info)))

info_list = info_list.append(info)

info.update(data_desc="data desc")

info_list

# ----------------------------------------------------------------------

import polars as pl
from datetime import date
from differences._OLD_NOW_PAGURO.info import _DataInfo

data = pl.DataFrame(
    {
        "numbers": [1, 2, 3],
        "letters": ["a", "b", "c"],
        "dates": [
            date(2010, 1, 1),
            date(2011, 1, 1),
            date(2012, 1, 1)]
    },
)

data_info = _DataInfo(data=data)
check = data_info.append(description={"data desc": "example data"})
