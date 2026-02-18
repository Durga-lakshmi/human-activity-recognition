from PIL import Image

img = Image.open("user01_exp00_bg.png").convert("RGB")
img.save("user01_exp00_bg.pdf", resolution=300)

img = Image.open("20260128_2220_after_counts.png").convert("RGB")
img.save("20260128_2220_after_counts.pdf", resolution=300)

img = Image.open("20260129_0752_after_prob.png").convert("RGB")
img.save("20260129_0752_after_prob.pdf", resolution=300)

img = Image.open("20260129_1850_CNN_TCN_before.png").convert("RGB")
img.save("20260129_1850_CNN_TCN_before.pdf", resolution=300)

img = Image.open("20260131_0214_CNN_TCN_tsne_test.png").convert("RGB")
img.save("20260131_0214_CNN_TCN_tsne_test.pdf", resolution=300)

img = Image.open("transition_prior.png").convert("RGB")
img.save("transition_prior.pdf", resolution=300)

#img = Image.open("user01_exp00_bg.png").convert("RGB")
#img.save("user01_exp00_bg.pdf", resolution=300)