from pydantic import BaseModel, Field
from typing import Literal

class Txn(BaseModel):
    step: int
    type: Literal["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"]
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrg: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
