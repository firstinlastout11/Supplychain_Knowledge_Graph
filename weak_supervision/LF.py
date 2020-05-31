from snorkel.labeling import labeling_function

POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1

# Check for the `spouse` words appearing between the person mentions
supplying = {"supplier", "supplied",  "supplying", "supplies", "supply"}
@labeling_function(resources=dict(supplying=supplying))
def lf_supply(row, supplying):
    for term in supplying:
        if term in row['sentence']:
             return POSITIVE
    return ABSTAIN

customer = {"customers","customer"}
@labeling_function(resources=dict(customer=customer))
def lf_customer(row, customer):
    for term in customer:
        if term in row['sentence']:
            return POSITIVE
    return ABSTAIN

sales_to = {"sales to"}
@labeling_function(resources=dict(sales_to=sales_to))
def lf_sales_to(row, sales_to):
    for term in sales_to:
        if term in row['sentence']:
            return POSITIVE
    return ABSTAIN

our_customer = {"our", "customers"}
@labeling_function(resources=dict(our_customer=our_customer))
def lf_our_customer(row, our_customer):
    if "our" in row['sentence'] and "customers" in row['sentence']:
        return POSITIVE
    return ABSTAIN

acquisition= {"acquisition", "acquired"}
@labeling_function(resources=dict(acquisition=acquisition))
def lf_acquisition(row, acquisition):
    for term in acquisition:
        if term in row['sentence']:
            return NEGATIVE
    return ABSTAIN

people = {"CEO",'ceo','manager','Manager','Mr.','Mrs.','Ms.'}
@labeling_function(resources=dict(people=people))
def lf_people(row, people):
    for term in people:
        if term in row['sentence']:
            return NEGATIVE
    return ABSTAIN

sold = {"sold to"}
@labeling_function(resources=dict(sold=sold))
def lf_sold(row, sold):
    for term in sold:
        if term in row['sentence']:
            return POSITIVE
    return ABSTAIN

relations = {"relationship","with"}
@labeling_function(resources=dict(relations=relations))
def lf_relation(row, relations):
    if "relation" in row['sentence'] and "with" in row['sentence']:
        return POSITIVE
    return ABSTAIN

competition = {"competitors","competition"}
@labeling_function(resources=dict(competition=competition))
def lf_competition(row, competition):
    for term in competition:
        if term in row['sentence']:
            return NEGATIVE
    return ABSTAIN