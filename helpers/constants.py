from enum import Enum


FRONT_END_URL = "http://localhost:3000"


class UserRoles(Enum):
    ADMIN = "admin"
    USER = "user"


class UserClusters(Enum):
    ADMIN_CLUSTER = "admin_cluster"
    UHCW_CLUSTER = "uhcw_cluster"
    PERSONAL_CLUSTER = "personal_cluster"
