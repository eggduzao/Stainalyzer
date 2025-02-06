
from datetime import date, datetime, timedelta
from random import seed, randint

seed(1987)

# Define the mapping of days to Portuguese
days_in_portuguese = ["Segunda", "Terça", "Quarta", 
                      "Quinta", "Sexta", "Sábado", "Domingo"]

# Start from the beginning of the current year
start_date = date(date.today().year, 1, 1)
end_date = date(date.today().year, 2, 28)

# Iterate through dates
current_date = start_date
while current_date <= end_date:

    day_of_week = date.weekday(current_date)
    dia_da_semana = days_in_portuguese[day_of_week]
    if(dia_da_semana not in ["Sábado", "Domingo"]):

        day = current_date.day    # Day as an integer
        month = current_date.month  # Month as an integer
        year = current_date.year   # Year as an integer
        day_diff = 0 if day-10 >= 0 else 1
        r1 = randint(8,11)
        r3 = randint(19,21)
        r2 = randint(0,59)
        r4 = randint(0,59)
        r1_diff = 0 if r1-10 >= 0 else 1
        r2_diff = 0 if r2-10 >= 0 else 1
        r4_diff = 0 if r4-10 >= 0 else 1

        print(f"{dia_da_semana} ({day:0{day_diff}}.0{month}.{year}):\t\t{r1:0{r1_diff}}:{r2:0{r2_diff}}-{r3}:{r4:0{r4_diff}}")

    current_date += timedelta(days=1)