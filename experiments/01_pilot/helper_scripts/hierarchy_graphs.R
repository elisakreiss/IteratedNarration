library(googleVis)
library(tidyverse)

# Flight
proposition_high = c("illegally flying")
proposition_medium = c('enter airport','first failed attempt','second successful attempt','being captured')
proposition_low_enter = c('hid her face','pass TSA','get screened')
proposition_low_first = c('pass gate agent','being stopped','directed to sit')
proposition_low_second = c('stayed overnight','shuttle to international gate','enter plane')
proposition_low_capture = c('detained by British costums','send back','charged')

prop_low_enter = data.frame("Proposition"=proposition_low_enter,"Parent"=c('enter airport'))
prop_low_first = data.frame("Proposition"=proposition_low_first,"Parent"=c('first failed attempt'))
prop_low_second = data.frame("Proposition"=proposition_low_second,"Parent"=c('second successful attempt'))
prop_low_capture = data.frame("Proposition"=proposition_low_capture,"Parent"=c('being captured'))
prop_medium = data.frame("Proposition"=proposition_medium,"Parent"=c('illegally flying'))
prop_high =  data.frame("Proposition"=proposition_high,"Parent"=NA)

df_flight = bind_rows(prop_low_enter,prop_low_first,prop_low_second,prop_low_capture,prop_medium,prop_high)
df_flight$Val = 0



Flight <- gvisOrgChart(df_flight, idvar = "Proposition", parentvar = "Parent", 
                     tipvar="Val", 
                     options=list(width=600, height=400,
                                  size='large', allowCollapse=TRUE))
plot(Flight)

# Bees
proposition_high = c("destruction of property")
proposition_medium = c('break-in','destruction','discovery','investigation')
# proposition_low_breakin = c('')
proposition_low_dest = c('overturn beehives','hack apart beehives','throw out equipment','smash equipment')
# proposition_low_disc = c('couple checks on 50 beehives','couple discovers damage','death of half a million bees','$60,000 property damage')
proposition_low_disc = c('couple checks on 50 beehives','couple discovers damage')
proposition_low_inv = c('capture','charge (in juvenile court)')

# prop_low_breakin = data.frame("Proposition"=proposition_low_breakin,"Parent"=c('break-in'))
prop_low_dest = data.frame("Proposition"=proposition_low_dest,"Parent"=c('destruction'))
prop_low_disc = data.frame("Proposition"=proposition_low_disc,"Parent"=c('discovery'))
prop_low_inv = data.frame("Proposition"=proposition_low_inv,"Parent"=c('investigation'))
prop_medium = data.frame("Proposition"=proposition_medium,"Parent"=c('destruction of property'))
prop_high =  data.frame("Proposition"=proposition_high,"Parent"=NA)

prop_level4_damage = data.frame("Proposition"=c('death of bees','property damage'),"Parent"=c('couple discovers damage'))
prop_level4_charge = data.frame("Proposition"=c('prison','damage fee'),"Parent"=c('charge (in juvenile court)'))

df_bees = bind_rows(prop_level4_damage,prop_level4_charge,prop_low_dest,prop_low_disc,prop_low_inv,prop_medium,prop_high)
df_bees$Val = 0

Bees <- gvisOrgChart(df_bees, idvar = "Proposition", parentvar = "Parent", 
                     tipvar="Val", 
                     options=list(width=600, height=400,
                                  size='large', allowCollapse=TRUE))
plot(Bees)
