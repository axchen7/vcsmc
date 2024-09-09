# importing geoopt the first time fails for some reason, but subsequent imports work
try:
    import geoopt
except:
    pass

from geoopt.manifolds import PoincareBall
