import pandas as pd

# Data
data = [
    (0, "macaron"), (1, "beignet"), (2, "cruller"), (3, "cockle_food"), (4, "samosa"), (5, "tiramisu"),
    (6, "tostada"), (7, "moussaka"), (8, "dumpling"), (9, "sashimi"), (10, "knish"), (11, "croquette"),
    (12, "couscous"), (13, "porridge"), (14, "stuffed_cabbage"), (15, "seaweed_salad"), (16, "chow_mein"),
    (17, "rigatoni"), (18, "beef_tartare"), (19, "cannoli"), (20, "foie_gras"), (21, "cupcake"),
    (22, "osso_buco"), (23, "pad_thai"), (24, "poutine"), (25, "ramen"), (26, "pulled_pork_sandwich"),
    (27, "bibimbap"), (28, "chicken_kiev"), (29, "apple_pie"), (30, "risotto"), (31, "fruitcake"),
    (32, "chop_suey"), (33, "haggis"), (34, "scrambled_eggs"), (35, "frittata"), (36, "scampi"),
    (37, "sushi"), (38, "orzo"), (39, "fritter"), (40, "nacho"), (41, "beef_stroganoff"), (42, "beef_wellington"),
    (43, "spring_roll"), (44, "savarin"), (45, "crayfish_food"), (46, "souffle"), (47, "adobo"),
    (48, "streusel"), (49, "deviled_egg"), (50, "escargot"), (51, "club_sandwich"), (52, "carrot_cake"),
    (53, "falafel"), (54, "farfalle"), (55, "terrine"), (56, "poached_egg"), (57, "gnocchi"),
    (58, "bubble_and_squeak"), (59, "egg_roll"), (60, "caprese_salad"), (61, "sauerkraut"), (62, "creme_brulee"),
    (63, "pavlova"), (64, "fondue"), (65, "scallop"), (66, "jambalaya"), (67, "tempura"), (68, "chocolate_cake"),
    (69, "potpie"), (70, "spaghetti_bolognese"), (71, "sukiyaki"), (72, "applesauce"), (73, "baklava"),
    (74, "salisbury_steak"), (75, "linguine"), (76, "edamame"), (77, "coq_au_vin"), (78, "tamale"),
    (79, "macaroni_and_cheese"), (80, "kedgeree"), (81, "garlic_bread"), (82, "beet_salad"),
    (83, "steak_tartare"), (84, "vermicelli"), (85, "pate"), (86, "pancake"), (87, "tetrazzini"),
    (88, "onion_rings"), (89, "red_velvet_cake"), (90, "compote"), (91, "lobster_food"), (92, "chicken_curry"),
    (93, "chicken_wing"), (94, "caesar_salad"), (95, "succotash"), (96, "hummus"), (97, "fish_and_chips"),
    (98, "lasagna"), (99, "lutefisk"), (100, "sloppy_joe"), (101, "gingerbread"), (102, "crab_cake"),
    (103, "sauerbraten"), (104, "peking_duck"), (105, "guacamole"), (106, "ham_sandwich"), (107, "crumpet"),
    (108, "taco"), (109, "strawberry_shortcake"), (110, "clam_chowder"), (111, "cottage_pie"),
    (112, "croque_madame"), (113, "french_onion_soup"), (114, "beef_carpaccio"), (115, "torte"), (116, "poi"),
    (117, "crab_food"), (118, "bacon_and_eggs"), (119, "coffee_cake"), (120, "custard"), (121, "syllabub"),
    (122, "pork_chop"), (123, "fried_rice"), (124, "boiled_egg"), (125, "galantine"), (126, "brisket"),
    (127, "reuben"), (128, "schnitzel"), (129, "ambrosia_food"), (130, "gyoza"), (131, "jerky"), (132, "ravioli"),
    (133, "fried_calamari"), (134, "spaghetti_carbonara"), (135, "miso_soup"), (136, "frozen_yogurt"),
    (137, "wonton"), (138, "panna_cotta"), (139, "french_toast"), (140, "enchilada"), (141, "ceviche"),
    (142, "fettuccine"), (143, "chili"), (144, "flan"), (145, "kabob"), (146, "sponge_cake"),
    (147, "casserole"), (148, "paella"), (149, "blancmange"), (150, "bruschetta"), (151, "tortellini"),
    (152, "grilled_salmon"), (153, "french_fries"), (154, "shrimp_and_grits"), (155, "churro"), (156, "donut"),
    (157, "meat_loaf_food"), (158, "meatball"), (159, "scrapple"), (160, "strudel"), (161, "coconut_cake"),
    (162, "marble_cake"), (163, "filet_mignon"), (164, "hamburger"), (165, "fried_egg"), (166, "tuna_tartare"),
    (167, "penne"), (168, "eggs_benedict"), (169, "bread_pudding"), (170, "takoyaki"), (171, "tenderloin"),
    (172, "chocolate_mousse"), (173, "baked_alaska"), (174, "hot_dog"), (175, "confit"), (176, "ham_and_eggs"),
    (177, "biryani"), (178, "greek_salad"), (179, "huevos_rancheros"), (180, "tagliatelle"),
    (181, "stuffed_peppers"), (182, "cannelloni"), (183, "pizza"), (184, "sausage_roll"), (185, "chicken_quesadilla"),
    (186, "hot_and_sour_soup"), (187, "prime_rib"), (188, "cheesecake"), (189, "limpet_food"), (190, "ziti"),
    (191, "mussel"), (192, "manicotti"), (193, "ice_cream"), (194, "waffle"), (195, "oyster"), (196, "omelette"),
    (197, "clam_food"), (198, "burrito"), (199, "roulade"), (200, "lobster_bisque"), (201, "grilled_cheese_sandwich"),
    (202, "gyro"), (203, "pound_cake"), (204, "pho"), (205, "lobster_roll_sandwich"), (206, "baby_back_rib"),
    (207, "tapenade"), (208, "pepper_steak"), (209, "welsh_rarebit"), (210, "pilaf"), (211, "dolmas"),
    (212, "coquilles_saint_jacques"), (213, "veal_cordon_bleu"), (214, "shirred_egg"), (215, "barbecued_wing"),
    (216, "lobster_thermidor"), (217, "steak_au_poivre"), (218, "huitre"), (219, "chiffon_cake"),
    (220, "profiterole"), (221, "toad_in_the_hole"), (222, "chicken_marengo"), (223, "victoria_sandwich"),
    (224, "tamale_pie"), (225, "boston_cream_pie"), (226, "fish_stick"), (227, "crumb_cake"),
    (228, "chicken_provencale"), (229, "vol_au_vent"), (230, "entrecote"), (231, "carbonnade_flamande"),
    (232, "bacon_lettuce_tomato_sandwich"), (233, "scotch_egg"), (234, "pirogi"), (235, "peach_melba"),
    (236, "upside_down_cake"), (237, "applesauce_cake"), (238, "rugulah"), (239, "rock_cake"),
    (240, "barbecued_spareribs"), (241, "beef_bourguignonne"), (242, "rissole"), (243, "mostaccioli"),
    (244, "apple_turnover"), (245, "matzo_ball"), (246, "chicken_cordon_bleu"), (247, "eccles_cake"),
    (248, "moo_goo_gai_pan"), (249, "buffalo_wing"), (250, "stuffed_tomato")
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Index", "Food"])

# Save to CSV
csv_path = "./ground_truth/foods_names.csv"
df.to_csv(csv_path, index=False)

csv_path