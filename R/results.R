library(tidyverse)
library(ggforestplot)
library(dmetar)
library(comprehenr)
library(grid)
library(showtext)
font_add_google(name = "Fira Sans", family="fira-sans")
# turn on showtext
showtext_auto()

df = read.csv("output/results-full.csv") %>% 
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.numeric, as.double) %>%
  as_tibble() %>%
  group_by(Test) %>%
  filter(Model != "igpt-logit") %>%
  mutate(p_min = min(p)) %>%
  mutate(N_a = n_a, N_t = n_t) %>%
  mutate(Model = str_replace(Model, "simclr", "SimCLRv2")) %>%
  mutate(Model = str_replace(Model, "igpt", "iGPT")) %>%
  arrange(Test) %>%
  ungroup()
df$se = se.from.p(effect.size = df$d, p=df$p, N=df$N_a, effect.size.type="difference")$StandardError
df$sig = factor(
  ifelse(df$p_min < 0.1, "bold(Significant)*phantom(' ')*(italic(p) < 0.1)", "bold(Insignificant)"),
  levels=c("bold(Significant)*phantom(' ')*(italic(p) < 0.1)", "bold(Insignificant)")
)

experiments = c(
  "Insect-Flower",
  "Gender-Science",
  "Gender-Career",
  "Skin-Tone",
  "Race",
  "Weapon",
  "Weapon (Modern)",
  "Native",
  "Asian",
  "Weight",
  "Religion",
  "Sexuality",
  "Disability",
  "Arab-Muslim",
  "Age"
)

cbp1 <- c("#56B4E9", "#009E73",
          "#0072B2", "#CC79A7")

main = df %>%
  filter(Test %in% experiments)
plot = forestplot(
  df = main,
  name = Test,
  estimate = d,
  pvalue = p,
  psignif = 0.1,
  ci= 0.9,
  colour = Model,
  xlab = "Effect Size (d)",
  ylab = "Bias Test (iEAT)",
  size=50,
  label = "label_parsed",
  fill = "transparent"
) +
  ggforce::facet_row(
    facets = ~sig,
    label = "label_parsed",
    scales = "free_y",
    space = "free"
  ) +
  scale_colour_manual(values=cbp1[3:4]) + 
  geom_text(
    label=paste0("N=",main$N_a+main$N_t),
    x=-7.5,
    vjust=2.5,
    size=3,
    fontface="italic"
  ) +
  coord_cartesian(clip='off') +
  theme(
    text=element_text(family="fira-sans")
  )
ggsave("output/results-abbrv.png", plot=plot, height=4, width=8, bg="transparent")


# Intersectional bias confusion matrices
intersectional = df %>%
  filter(grepl("Intersectional", Test)) %>%
  mutate(d = ifelse(p < 0.1, d, NA))

valencerace = intersectional %>%
  filter(grepl("Intersectional-Valence", Test), Model == "iGPT") %>%
  filter(grepl("white-", X) & grepl("black-", Y))

fillscale = scale_fill_distiller("Effect size (d)", limits=c(0, 2), palette="Blues", direction=1, na.value="grey50")
wb_bias = 1.16
ggplot(data=valencerace, mapping=aes(x=X, y=Y)) +
  geom_tile(aes(fill = d), alpha=ifelse(is.na(valencerace$d), 0, 1)) +
  geom_text(aes(label=ifelse(is.na(d), "insignificant", round(d,2)))) +
  fillscale +
  scale_x_discrete("White (Pleasant)", labels=c("White Woman", "White Man")) +
  scale_y_discrete("Black (Unpleasant)", labels=c("Black Woman", "Black Man")) +
  theme(
    text=element_text(family="fira-sans")
  ) +
  ggtitle("White/Black vs. Pleasant/Unpleasant") +
  coord_cartesian(clip="off") +
  annotate("rect", xmin=0.8, xmax=2.2, ymin=1.35, ymax=1.65, fill=fillscale$map(wb_bias)) +
  annotate("text", x=1.5, y=1.5, label=paste0("Overall Race Bias\n", wb_bias)) +
  theme_minimal()
ggsave("output/intersectional_valence-race.png", height=4, width=5, bg="transparent")

valencegender = intersectional %>%
  filter(grepl("Intersectional-Valence", Test), Model == "iGPT") %>%
  add_row(X="black-female", Y="white-male", d=-0.827) %>%
  filter(grepl("-female", X) & grepl("-male", Y))

fm_bias = 0.39
fillscale = scale_fill_distiller("Effect size (d)", limits=c(-2, 2), palette="RdBu", direction=1, na.value="grey50")
ggplot(data=valencegender, mapping=aes(x=X, y=Y)) +
  geom_tile(aes(fill = d), alpha=ifelse(is.na(valencegender$d), 0, 1)) +
  geom_text(aes(label=ifelse(is.na(d), "insignificant", round(d,2)))) +
  fillscale +
  scale_x_discrete("Woman (Pleasant)", labels=c("Black Woman", "White Woman")) +
  scale_y_discrete("Man (Unpleasant)", labels=c("Black Man", "White Man")) +
  theme(
    text=element_text(family="fira-sans")
  ) +
  ggtitle("Woman/Man vs. Pleasant/Unpleasant") +
  coord_cartesian(clip="off") +
  annotate("rect", xmin=0.8, xmax=2.2, ymin=1.35, ymax=1.65, fill=fillscale$map(fm_bias)) +
  annotate("text", x=1.5, y=1.5, label=paste0("Overall Gender Bias\n", fm_bias)) +
  theme_minimal()
ggsave("output/intersectional_valence-gender.png", height=4, width=5, bg="transparent")

gendercareer = intersectional %>%
  filter(grepl("Intersectional-Gender-Career", Test), Model == "iGPT") %>%
  filter(grepl("-female", Y))

fm_bias = 0.81
fillscale = scale_fill_distiller("Effect size (d)", limits=c(0, 2), palette="Blues", direction=1, na.value="grey50")
ggplot(data=gendercareer, mapping=aes(x=X, y=Y)) +
  geom_tile(aes(fill = d), alpha=ifelse(is.na(gendercareer$d), 0, 1)) +
  geom_text(aes(label=ifelse(is.na(d), "insignificant", round(d,2)))) +
  fillscale +
  scale_x_discrete("Man (Career)", labels=c("Black Man", "White Man")) +
  scale_y_discrete("Woman (Family)", labels=c("Black Woman", "White Woman")) +
  theme(
    text=element_text(family="fira-sans")
  ) +
  ggtitle("Man/Woman vs. Career/Family") +
  coord_cartesian(clip="off") +
  annotate("rect", xmin=0.8, xmax=2.2, ymin=1.35, ymax=1.65, fill=fillscale$map(fm_bias)) +
  annotate("text", x=1.5, y=1.5, label=paste0("Overall Gender Bias\n", fm_bias)) +
  theme_minimal()
ggsave("output/intersectional_gender-career.png", height=4, width=5, bg="transparent")

