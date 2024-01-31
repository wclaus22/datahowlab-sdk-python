from pydantic import BaseModel, Field
from dhl_sdk.client import Client
from dhl_sdk.crud import CRUDClient

from dhl_sdk.entities import Variable


PRODUCTS_URL = "api/db/v2/products"
RECIPES_URL = "api/db/v2/recipes"
FILES_URL = "api/db/v2/files"
EXPERIMENTS_URL = "api/db/v2/experiments"


class Product(BaseModel):
    """Pydantic model for Product"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    tags: list = Field(alias="tags")
    code: str = Field(alias="code")

    @staticmethod
    def requests(client: Client) -> CRUDClient["Product"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Variable"](client, PRODUCTS_URL, Product)


class Recipe(BaseModel):
    """Pydantic model for Recipe"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    tags: list = Field(alias="tags")
    product: Product = Field(alias="product")
    parent_id: str = Field(alias="parentId")
    variables: list[Variable] = Field(alias="variables")

    @staticmethod
    def requests(client: Client) -> CRUDClient["Recipe"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Recipe"](client, RECIPES_URL, Recipe)


class File(BaseModel):
    """Pydantic model for File"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    tags: list = Field(alias="tags")
    type: str = Field(alias="type")

    @staticmethod
    def requests(client: Client) -> CRUDClient["File"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["File"](client, FILES_URL, File)


class Experiment(BaseModel):
    """Pydantic model for Experiment"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    tags: list = Field(alias="tags")
    product: Product = Field(alias="product")
    variables: list[Variable] = Field(alias="variables")

    @staticmethod
    def requests(client: Client) -> CRUDClient["Experiment"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Experiment"](client, EXPERIMENTS_URL, Experiment)
