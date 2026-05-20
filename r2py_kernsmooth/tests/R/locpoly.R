library(KernSmooth)
library(carData)

income   <- Prestige$income
prestige <- Prestige$prestige

result_truncate <- locpoly(income, prestige, bandwidth = 5000)
print(result_truncate)

result_no_truncate <- locpoly(income, prestige, bandwidth = 5000, truncate = FALSE)
print(result_no_truncate)
