# Kaggle首戰斬獲第三，看深度學習菜鳥團隊如何一鳴驚人。
南華大學跨領域-人工智慧期中報告
11223025林彥融
# 從一個簡單易懂的深度學習指南開始

不要想著馬上就能理解所有東西，這需要大量的練習。本指南旨在向深度學習初學者展示 fast.ai 的魅力。假定你了解一些 python 知識，也對機器學習稍有涉獵。這樣的話，我們就走上了學習正軌。

（引用）本文展示的所有程式碼可在 Google Colaboratory 中找到：這是一個 Jupyter 筆記本環境，不需要進行任何設定就可以使用，並且完全在雲端運行。你可以透過 Colaboratory 編寫和執行程式碼，保存和分享分析，存取大量的運算資源，所有這些都是免費的。

程式碼請參閱：https://colab.research.google.com/drive/1PVaRPY1XZuPLtm01V2XxIWqhLrz3_rgX。

# 導入 fast.ai 和我們將要使用的其他函式庫

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_123317_www.sohu.com.jpeg)

輸入庫

# 取得競賽數據

為了盡可能簡潔明了，Abdishakur 上傳競賽資料檔至 dropbox.com。你可以在競賽頁面上找到這些數據。你需要接受競賽規則，並在參賽後存取資料。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_123747_www.sohu.com.jpeg)

# 觀察數據

我們在解決一個問題時首先要做的是觀察可用資料。在想出解決方案之前，我們需要先理解這個問題以及資料是什麼樣的。觀察資料意味著理解資料目錄的構成方式、資料標籤以及樣本影像是什麼樣的。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124210_www.sohu.com.jpeg)

使用 pandas 庫來讀取資料。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124438_www.sohu.com.jpeg)

訓練模型所要使用的資料標籤。


處理「影像分類資料集」和「表格資料集」的主要區別在於標籤的儲存方式。這裡的標籤指的是圖像中的內容。在這個特定的資料集中，標籤以 CSV 檔案格式儲存。

想要了解更多計算「分數」欄位的方法，點選：

https://success.figure-eight.com/hc/en-us/articles/201855939-How-to-Calculate-a-Confidence-Score。

我們將使用 seaborn 的 countplot 函數來觀察訓練資料的分佈。我們從下圖中看到，大約 14300 個圖像中沒有發現油棕種植園，而僅有 942 個圖像中發現了油棕種植園。這就是所謂的不平衡資料集，但我們在這裡不討論這個深度學習問題。我們此刻正邁出了一小步。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124753_www.sohu.com.jpeg)

統計兩個類別的樣本數。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_124955_www.sohu.com.jpeg)

訓練資料集中分佈

#準備數據

提供的測試資料放置於兩個不同的資料夾：leaderboard 留出資料和 leaderboard 測試資料。由於競賽要求提交這兩種資料集的預測，所以我們將兩者結合。我們共獲得 6534 副圖像。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_125333_www.sohu.com.jpeg)

結合 leaderboard 留出資料和 leaderboard 測試資料。

我們將使用 fast.ai 的 DataBlock API 來構成數據，這是將資料集呈現給模型的簡單方法。

![imagr](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_125522_www.sohu.com.jpeg)

建立一個 ImageList 來保留數據

我們將使用 ImageList 來保存訓練數據，並使用 from_df 方法讀取資料。這樣做的原因是，我們將訓練集資訊儲存在了名為 df 的 DataFrame 中。

接下來需要隨機分割訓練集，並保留 20% 作為驗證集，以便在訓練中監督模型表現。我們選擇了一個 seed，確保再一次訓練時能得到相同的結果，透過相同的 seed，我們就能知道哪些改進是好的，哪些是壞的。

此外，我們同樣還要把訓練集的標籤位址提供給 ImageList，並將資料與標籤合併。

最後，還需要在資料上執行轉換，透過設定 flip_vert = True 將翻轉影像，這能幫助模型辨識不同朝向的影像。此外，還需要使用 imagenet_stats 來歸一化映像。


# 預覽影像

如下是有或沒有油棕種植園的衛星圖：

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_131638_www.sohu.com.jpeg)

展示兩個 Batch 的圖像。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_131827_www.sohu.com.jpeg)

有油棕的圖像標記為 1，沒有油棕的標記為 0

# 訓練我們的模型

現在，開始訓練我們的模型。我們將使用卷積神經網路作為主體，並利用 ResNet 模型的預訓練權重。 ResNet 模型被訓練用來對各種影像進行分類，不用擔心它的理論和實作細節。現在，我們建立的模型以衛星圖像作為輸入，並輸出這兩個類別的預測機率。

卷積神經網絡

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_132244_www.sohu.com.jpeg)

搜尋最佳模型學習率。

接下來，我們用 lr_find() 函數找到了理想的學習率，並使用 recorder.plot() 進行了視覺化。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_132851_www.sohu.com.jpeg)

搜尋最佳模型學習率。

我們將選擇斜率最大的學習率，在這裡我們選擇的是 1e-2。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_133040_www.sohu.com.jpeg)

以學習率 1e-2 對模型展開 5 個週期的訓練。

我們將使用 fit_one_cycle 函數對模型進行 5 個 epoch 的訓練（遍歷所有資料 5 次）。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_133253_www.sohu.com.jpeg)

訓練和驗證損失。

注意展示的 metrics，即 training_loss 和 valid_loss。隨著時間的推移，我們使用它們來監控模型的改進。

最佳模型是在第四個 epoch 時獲得的。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_133531_www.sohu.com.jpeg)

訓練階段模型的輸出；訓練和驗證損失的變化過程。

當你進行訓練和驗證資料集時，fast.ai 只在內部挑選並保存你的最佳模型。

# 評估我們的模型

競賽提交的資料是根據預測機率和觀測目標 has_oilpalm 之間的 ROC 曲線來評估的。預設情況下，Fast.ai 不會附帶這個指標，所以我們將使用 scikit-learn 函式庫。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_133718_www.sohu.com.jpeg)

列印驗證指標。

使用預訓練模型和 fast.ai 的妙處在於，你可以獲得很好的預測準確率。在我們的案例中，沒有費多大力就獲得了 99.44% 的準確率。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_133914_www.sohu.com.jpeg)

訓練第一階段的指標。

儲存模型，並繪製關於預測的混淆矩陣。

```
learn.save('resnet50-stgl')
```

# 使用混淆矩陣查看結果

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_134341_www.sohu.com.jpeg)

繪製混淆矩陣

混淆矩陣是一種圖形化的方式，可以查看模型準確或不準確的預測影像數量。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_134545_www.sohu.com.jpeg)

第一個訓練階段的混淆矩陣。

從這個矩陣可以看出，模型準確地預測有 2863 張影像中沒有油棕，168 張影像中有油棕。 10 張有油棕的圖像被分類為沒有，而 7 張沒有油棕的圖像則被分類為有油棕。

對這種簡單的模型來說，這個結果不錯了。接下來，我們搜尋到了訓練的理想學習率。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_134813_www.sohu.com.jpeg)

搜尋理想的學習率。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_134948_www.sohu.com.jpeg)

我們在學習率 1e-6 和 1e-4 之間選擇了一個學習率。

在 7 個 epoch 內，使用 1e-6 和 1e-4 之間的最大學習率來擬合模型。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_135134_www.sohu.com.jpeg)

對模型進行 7 個週期的訓練，學習率應在 1e-6 和 1e-4 範圍內。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_135329_www.sohu.com.jpeg)

訓練和驗證損失。

以圖形方式觀察訓練指標，以監控每個訓練週期後模型的表現。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_135653_www.sohu.com.jpeg)

訓練階段模型的輸出；訓練和驗證損失的變化過程。

保存模型的第二個訓練階段：

```
learn.save('resnet50-stg2')
```

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_135946_www.sohu.com.jpeg)

準確率、誤差率和 AUC 分數

列印模型的準確率、誤差率和 AUC 指標：

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14152_www.sohu.com.jpeg)

第二個訓練階段的指標。

如你所見，模型的準確率從 99.44% 上升到了 99.48%。誤差率從 0.0056 降到了 0.0052。 AUC 也從 99.82% 上升為 99.87%。

![imagr](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14523_www.sohu.com.jpeg)

繪製混淆矩陣。

與我們繪製的上一個混淆矩陣相比，你會發現模型的預測效果更好了。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14829_www.sohu.com.jpeg)
第二個訓練階段的混淆指標

之前有 7 張不含油棕種植園的圖像被錯誤分類，現在降到了 3 張，這就是進步。

我們在訓練和調參期間遵循了一種模式。大多數深度學習實驗都遵循類似的迭代模式。

# 影像轉換

我們將在數據上執行更多的影像轉換，這應該是能提升模型效果的。影像轉換的具體描述可以在 fast.ai 文件中找到：

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14108_www.sohu.com.jpeg)

應用不同的轉換以提升模型效果

max_lighting：如果超參不為 None，則以 p_lighting 為機率隨機進行亮度、對比的調整，且最大亮度不超過 max_lighting。

max_zoom：如果超參不小於 1，那麼以 p_affine 為機率隨機放大 1 到 max_zoom 倍。

max_warp：如果超參不為 None，那麼以 p_affine 為機率在-max_warp 和 max_warp 之間隨機對稱變換。

我們再一次搜尋最優學習率：

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_141143_www.sohu.com.jpeg)

搜尋一個合理的學習率

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14223_www.sohu.com.jpeg)

我們選擇的學習率是 1e-6

將模型訓練 5 個週期。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14240_www.sohu.com.jpeg)

訓練 5 個週期

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_142551_www.sohu.com.jpeg)

訓練和驗證損失

比較訓練指標，並與過去的指標進行比較。我們的模型在這次迭代中略遜於 0.0169 和 0.0163。先不要洩氣。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_142740_www.sohu.com.jpeg)

訓練階段模型的輸出；在第 3 個 epoch 時得到最佳模型

儲存模型訓練的第三個階段並列印出指標。如圖所示，目前模型的準確率為 99.38，上一個階段的準確率為 99.48%。 AUC 分數從 99.87% 提高到了 99.91%，這是比賽評分的標準。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_142940_www.sohu.com.jpeg)

準確率、誤差率和 AUC 分數

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_143239_www.sohu.com.jpeg)

第三個訓練階段的指標


# 最終訓練階段

你可能注意到了，我們剛開始使用的圖像大小為 164，然後逐漸增加到 256（如下）。這麼做是為了利用 fast.ai 用於分類的漸進式圖像大小縮放，即在一開始使用小圖像，之後隨著訓練逐漸增加圖像大小。如此一來，當模型早期非常不準確時，它能迅速看到大量圖像並實現快速改進，而在後期訓練中，它可以看到更大的圖像，學到更多細粒度的差別。 （詳情請參閱：現在，所有人都可以在 18 分鐘內訓練 ImageNet 了）

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_143528_www.sohu.com.jpeg)

應用不同的變換來改進模型，將圖像大小增加到 256

我們又發現了一個最佳學習率。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_14393_www.sohu.com.jpeg)

找到理想學習率

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_144256_www.sohu.com.jpeg)

找到理想的學習率

以 1e-4 的學習率訓練 5 個 epoch 以擬合模型。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_144511_www.sohu.com.jpeg)

以 1e-4 的學習率對模型訓練 5 個週期

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_144656_www.sohu.com.jpeg)

訓練和驗證損失

觀察訓練指標並與先前的指標比較。我們的模型有了小小的提升（損失從 0.169 降到了 0.168）。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_144842_www.sohu.com.jpeg)

模型訓練階段的輸出。在第 2 個 epoch 時得到最佳模型

儲存模型最後的訓練階段並列印出指標。

```
learn.save('resnet50-stg4')
```

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_145137_www.sohu.com.jpeg)

準確率、誤差率和 AUC 分數

如下所示，模型的準確率為 99.44%，優於上一個訓練階段 99.38% 的準確率。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_145321_www.sohu.com.jpeg)

第四個訓練階段的指標

準備一個競賽提交文件

現在可以看到我們的模型對未見過的數據做出了多麼好的預測。

![image](https://github.com/jacky5649/1221/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_21-12-2024_145453_www.sohu.com.jpeg)

準備一個 CSV 提交文件

# 將文件提交給 WiDS Datathon

你仍然可以參加 WiDS 競賽並晚一點提交。進入參賽頁面，點選“Join Competition”，了解比賽規則。現在你可以提交作品，看看自己會排到第幾。

根據模型預測對提交的作品進行評分

原文連結：https://towardsdatascience.com/how-a-team-of-deep-learning-newbies-came-3rd-place-in-a-kaggle-contest-644adcc143c8
