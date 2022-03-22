import csv
import matplotlib.pyplot as plt
Rand_round = []
Rand_steps = []
Rand_score = []
with open('learning_stat_random.csv', newline='') as csvfile:
    # reader = csv.reader(csvfile, delimiter=',')
    # for row in reader:
    #   List.append( [(', '.join(row))])
   reader = csv.DictReader(csvfile)
   for row in reader:
       
         Rand_round.append(int(row['round']))
         Rand_steps.append(int(row['steps']))
         Rand_score.append(int(row['score']))


opt_round = []
opt_steps = []
opt_score = []
with open('learning_stat_opt.csv', newline='') as csvfile:
    # reader = csv.reader(csvfile, delimiter=',')
    # for row in reader:
    #   List.append( [(', '.join(row))])
   reader = csv.DictReader(csvfile)
   for row in reader:
       
         opt_round.append(int(row['round']))
         opt_steps.append(int(row['steps']))
         opt_score.append(int(row['score']))

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Scoure agent achieved in respective round')

ax1.plot(Rand_round,Rand_score, 'b')
#ax1.set_title('Random initialization')
ax1.set(xlabel='# rounds', ylabel='# steps')
ax2.plot(opt_round,opt_score, 'r')
#ax2.set_title('Optmised initialization')
ax2.set(xlabel='# rounds', ylabel='# steps')


