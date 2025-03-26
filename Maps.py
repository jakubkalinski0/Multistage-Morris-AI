#wieżchołki są numerowane od lewego górnego rogu od zera

THREE_MEN_GRAPH = ((1,3,4),(0,2,4),(1,4,5),
                   (0,4,6),(0,1,2,3,5,6,7,8),(2,4,8),
                   (3,4,7),(4,6,8),(4,5,7))

THREE_MEN_MILLS= ((0,1,2),(3,4,5),(6,7,8),
                    (0,3,6),(1,4,7),(2,5,8),
                    (0,4,8),(2,4,6))

SIX_MEN_GRAPH = ((1,6),(0,2,4),(1,9),
                 (4,7),(1,3,5),(4,8),
                 (0,7,13),(3,6,10),(5,9,12),(2,8,15),
                 (7,11),(10,12,14),(8,11),
                 (6,14),(11,13,15),(9,14))

SIX_MEN_MILLS = ((0,1,2),(3,4,5),(10,11,12),(13,14,15),
                  (0,6,13),(3,7,10),(5,8,12),(2,9,15))


NINE_MEN_GRAPH = ((1,9),(0,2,4),(1,14),
                  (4,10),(1,3,5,7),(4,13),
                  (7,11),(4,6,8),(7,12),
                  (0,10,21),(3,9,11,18),(6,10,15),(8,13,17),(5,12,14,10),(2,13,23),
                  (11,16),(15,17,19),(12,16),
                  (10,19),(16,18,20,22),(13,19),
                  (9,22),(19,21,23),(14,22))

NINE_MEN_MILLS = ((0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(15,16,17),(18,19,20),(21,22,23),
                   (0,9,21),(3,10,18),(6,11,15),(8,12,17),(5,13,20),(2,14,23),
                     (1,4,7),(16,19,22))

TWELVE_MEN_GRAPH=((1,9,3),(0,2,4),(1,14,5),
                  (4,10,0,6),(1,3,5,7),(4,13,2,8),
                  (7,11,3),(4,6,8),(7,12,5),
                  (0,10,21),(3,9,11,18),(6,10,15),(8,13,17),(5,12,14,10),(2,13,23),
                  (11,16,18),(15,17,19),(12,16,20),
                  (10,19,15,21),(16,18,20,22),(13,19,17,23),
                  (9,22,18),(19,21,23),(14,22,20))

TWELVE_MEN_MILLS =((0,1,2),(3,4,5),(6,7,8),(9,10,11),(12,13,14),(15,16,17),(18,19,20),(21,22,23),
                  (0,9,21),(3,10,18),(6,11,15),(8,12,17),(5,13,20),(2,14,23),
                  (1,4,7),(16,19,22)
                  (0, 3, 6),(2, 5, 8),(15, 18, 21),(17, 20, 23))