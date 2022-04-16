# パッケージの読み込み --------------------------------------------------------------

library(tidymodels) # モデル構築
library(tidyverse) # とりあえず
library(tidylog) # 処理のログが出て嬉しい

library(DALEX) # モデルの解釈
library(sf) # 地図可視化

library(glue) # 変数を文字列にいれられる
library(skimr) # データ確認用

set.seed(42) # 同じ結果が出るように

# データの読み込み----------------------------------------------------------------

# Ames Housing Data
# データの説明：http://jse.amstat.org/v19n3/decock/DataDocumentation.txt
data(ames, package = "modeldata")

# 必要な地図情報をダウンロードして展開する
# https://mapcruzin.com/free-united-states-shapefiles/free-iowa-arcgis-maps-shapefiles.htm
download_iowa_highway <- function(target) {
  # 地図情報をダウンロード
  download.file(
    url = glue("https://mapcruzin.com/download-shapefile/us/{target}.zip"),
    destfile = "iowa_highway.zip"
  )

  # zip fileを解凍
  unzip(zipfile = glue("{target}.zip"), exdir = target)

  # 元のzip fileを削除
  file.remove(glue("{target}.zip"))
}
download_iowa_highway(target = "iowa_highway")

iowa_highway <- st_read(dsn = "iowa_highway/iowa_highway.shp")


# データの確認-------------------------------------------------------------------------

ames
ames %>% glimpse()
ames %>% summary()
ames %>% skimr::skim()

# 目的変数の分布を確認
plot_histogram <- function(df, var) {
  df %>%
    ggplot(aes({{ var }})) +
    geom_histogram(color = "white", fill = "gray30")
}

# 通常ver
ames %>%
  plot_histogram(Sale_Price)

# 対数ver
ames %>%
  plot_histogram(log(Sale_Price))


# 目的変数と説明変数の関係を確認
plot_scatter <- function(df, x_var, y_var) {
  df %>%
    ggplot(aes({{ x_var }}, {{ y_var }})) +
    geom_point(color = "gray30", alpha = 0.3) +
    geom_smooth()
}

# 家の広さ
ames %>%
  plot_scatter(Gr_Liv_Area, Sale_Price)

# 家の広さ対数ver
ames %>%
  plot_scatter(Gr_Liv_Area, log(Sale_Price))

# 家の広さ対数-対数ver
ames %>%
  plot_scatter(log(Gr_Liv_Area), log(Sale_Price))

# 築年数
ames %>%
  plot_scatter(Year_Built, Sale_Price)

# 築年数対数ver
ames %>%
  plot_scatter(Year_Built, log(Sale_Price))


# 緯度経度と目的変数の関係を確認
plot_map <- function(df, map_info, color_var) {
  # 地図の描画範囲を制限
  x_lims <- extendrange(df$Longitude)
  y_lims <- extendrange(df$Latitude)

  df %>%
    ggplot() +
    geom_sf(aes(geometry = geometry), data = iowa_highway, color = "gray") +
    geom_point(aes(Longitude, Latitude, color = {{ color_var }})) +
    lims(x = x_lims, y = y_lims) +
    scale_color_viridis_c() +
    theme_void()
}

ames %>%
  plot_map(iowa_highway, log(Sale_Price))


# データの分割 ------------------------------------------------------------------

# 訓練データとテストデータに分割
train_test_split <- rsample::initial_split(
  ames,
  prop = 3 / 4, # 訓練データの割合
  strata = "Sale_Price" # 一応目的変数の分布がずれないようにしておく
)

train_test_split

df_train <- rsample::training(train_test_split)
df_test <- rsample::testing(train_test_split)

df_train

df_test


# 前処理 ---------------------------------------------------------------------

# 目的変数の対数だけとる
rec <- recipes::recipe(Sale_Price ~ ., data = df_train) %>%
  step_log(Sale_Price)


# モデルの定義 -----------------------------------------------------------

# Random Forestを利用。ハイパーパラメータいじらなくてもそれなりに精度がいいので
model <- parsnip::rand_forest() %>%
  parsnip::set_engine(
    engin = "ranger",
    num.threads = parallel::detectCores(), # 処理を全コアで分散
    seed = 42
  ) %>%
  parsnip::set_mode("regression") # 分類なら"classification"


# ワークフローの作成-------------------------------------------------------------------------

# 前処理をしたrecipeがあるならそれもadd_recipe()で追加する
wf <- workflows::workflow() %>%
  workflows::add_model(model) %>% # モデルの指定
  workflows::add_recipe(rec)


# モデルの学習と予測 ---------------------------------------------------------------

# last_fit()は全訓練データで学習して、テストデータで予測する
# 本来はCVでハイパーパラメータをチューニングしてから使うが、便利なので
first_result <- wf %>%
  tune::last_fit(split = train_test_split)

first_result

# テストデータでの予測精度を確認
first_result %>%
  tune::collect_metrics()

# テストデータに対する予測結果
first_pred <- first_result %>%
  tune::collect_predictions()

first_pred

# 予測値と実測値を比較
plot_pred_actual <- function(df, actual_var) {
  lims <- extendrange(pull(df, {{ actual_var }}))
  df %>%
    ggplot(aes(x = {{ actual_var }}, y = .pred)) +
    geom_abline(color = "gray", size = 1) +
    geom_point(alpha = 0.3) +
    coord_fixed(xlim = lims, ylim = lims) +
    labs(x = "Actual", y = "Prediction")
}

first_pred %>%
  plot_pred_actual(Sale_Price)


# Cross Validation --------------------------------------------------------

# 訓練データを、さらに訓練データとバリデーションデータに分割
# 4 fold cross validation
cv_split <- rsample::vfold_cv(
  df_train,
  v = 4, # 何分割するか
  strata = "Sale_Price"
)

cv_split

# 学習してバリデーションデータでの予測精度を確認
cv_result <- wf %>%
  tune::fit_resamples(resamples = cv_split)

cv_result

cv_result %>%
  tune::collect_metrics()


# ハイパーパラメータのチューニング --------------------------------------------------------

# Random Forestの重要なハイパーパラメータは3つ
# treesはある程度多ければいい。あんまり多いと重くなるので1000で決め打ち
# min_nとmtryをチューニングする。チューニングしたいパラメータをtune()としておく
model_tune <- model %>%
  # 元のモデルをベースにアップデートできる
  update(
    trees = 1000,
    min_n = tune::tune(),
    mtry = tune::tune()
  )

# パラメータの探索範囲
tune_parameters <- tune::parameters(
  list(
    min_n = dials::min_n(range = c(1, 10)),
    mtry = dials::mtry(range = c(1, ncol(df_test) - 1))
  )
)

# 元のワークフローをベースにアップデートできる
wf_tune <- wf %>%
  workflows::update_model(model_tune)

# ハイパーパラメータの探索
# tune_bayes()ならベイズ最適化、tune_grid()でグリッドサーチもできる
bayes_result <- wf_tune %>%
  tune::tune_bayes(
    resamples = cv_split,
    param_info = tune_parameters,
    metrics = yardstick::metric_set(rmse), # RMSEが一番良くなるパラメータを探す
    initial = 5,
    iter = 30,
    control = tune::control_bayes(verbose = TRUE, no_improve = 5)
  )

bayes_result

bayes_result %>%
  tune::collect_metrics()

bayes_result %>%
  tune::show_best()

best_model <- bayes_result %>%
  tune::select_best()


# 一番良かったモデルで最終結果を確認 -------------------------------------------------------

# 一番良かったモデルにアップデート
wf_final <-
  wf_tune %>%
  tune::finalize_workflow(best_model)

# 全訓練データで学習して、テストデータで予測
final_result <- wf_final %>%
  tune::last_fit(train_test_split)

# 予測精度を確認
final_result %>%
  tune::collect_metrics()

# 最初のモデルの予測精度と比較
first_result %>%
  tune::collect_metrics()


# モデルを解釈 ------------------------------------------------------------------

# 解釈に利用するデータと目的変数を指定する
explainer <- final_result %>%
  workflows::extract_fit_parsnip() %>% # モデルを取り出す
  DALEX::explain(
    data = df_test %>% select(!Sale_Price),
    y = log(df_test$Sale_Price),
    label = "Random Forest"
  )

## Permutation Feature Importance ------------------------------------------------------
# 説明変数の重要度
# Bはシャッフルの回数。重いので1にした。データ量によるが、10回くらいやったほうがいいと思う
pfi <- explainer %>%
  DALEX::model_parts(B = 1)

pfi %>%
  plot(max_vars = 10)

## Partial Dependence ------------------------------------------------------
# 説明変数と目的変数の平均的な関係
target_variable <- "Year_Built"

pd <- explainer %>%
  DALEX::model_profile(variables = target_variable)

pd %>%
  plot()

## Individual Conditional Expectation ------------------------------------------------------
# 説明変数と目的変数のインスタンスごとの関係
ice <- explainer %>%
  DALEX::predict_profile(
    new_observation = df_test %>% sample_n(10),
    variables = target_variable
  )

ice %>%
  plot(variables = target_variable)

## breakdown ------------------------------------------------------
# インスタンスごとの予測値に対する各説明変数の貢献度を分解

# type = "shap"とするとSHapley Additive exPlanationsになるが、めっちゃ重い
bd <- explainer %>%
  DALEX::predict_parts(
    new_observation = df_test %>% sample_n(1),
    type = "break_down"
  )

bd %>%
  plot(max_vars = 10)
