import csv
import matplotlib.pyplot as plt

loss_log_path = "outputs/loss_log.csv"

epochs = []
lossG = []
lossD = []

# อ่านข้อมูลจากไฟล์ CSV
with open(loss_log_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        lossG.append(float(row["lossG"]))
        lossD.append(float(row["lossD"]))

# วาดกราฟ
plt.figure(figsize=(10, 6))
plt.plot(epochs, lossG, label="Generator Loss (lossG)", color="red")
plt.plot(epochs, lossD, label="Discriminator Loss (lossD)", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_plot.png")
plt.show()