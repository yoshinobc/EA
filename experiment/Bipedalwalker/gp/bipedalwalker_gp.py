import numpy as np
from deap import algorithms,base,creator,tools,gp
import random
import operator
import math
import gym
from gym import wrappers
import time
import  networkx as nx
import matplotlib.pyplot as plt
import itertools
i = 0
ENV = gym.make("BipedalWalker-v2")
MAX_STEPS=3600
NGEN=300

def sigmoid(num):
    return 1 / (1 + np.exp(num))

def relu(num):
    return np.maximum(0,num)

def safeDiv(left,right):
    if(right==0):
        return 0
    try:
        return left/right
    except ZeroDivisionError:
        return 0

def clip2(num):
    return np.clip(num,-1,1)

def end(inp1,inp2,inp3,inp4):
    return [inp1,inp2,inp3,inp4]

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float,24),list,"ARG")
pset.addPrimitive(operator.add,[float,float],float)
pset.addPrimitive(operator.sub,[float,float],float)
pset.addPrimitive(operator.mul,[float,float],float)
pset.addPrimitive(safeDiv,[float,float],float)
pset.addPrimitive(operator.neg,[float],float)
pset.addPrimitive(math.cos,[float],float)
pset.addPrimitive(math.sin,[float],float)
pset.addPrimitive(sigmoid,[float],float)
pset.addPrimitive(relu,[float],float)
pset.addPrimitive(clip2,[float],float)
pset.addPrimitive(end,[float,float,float,float],list)
pset.addEphemeralConstant("rand101", lambda:random.randint(-1,1)*100,float)

pset.renameArguments(ARG0='hull_angle')
pset.renameArguments(ARG1='hull_angularVelocity')
pset.renameArguments(ARG2='vel_x')
pset.renameArguments(ARG3='vel_y')
pset.renameArguments(ARG4='hip_joiint_1_angle')
pset.renameArguments(ARG5='hip_joint_1_speed')
pset.renameArguments(ARG6='knee_joint_1_angle')
pset.renameArguments(ARG7='knee_joint_1_speed')
pset.renameArguments(ARG8='leg_1_ground_contact_flag')
pset.renameArguments(ARG9='hip_joint_2_angle')
pset.renameArguments(ARG10='hip_joint_2_speed')
pset.renameArguments(ARG11='knee_joint_2_angle')
pset.renameArguments(ARG12='knee_joint_2_speed')
pset.renameArguments(ARG13='leg_2_ground_contact_flag')
pset.renameArguments(ARG14='lidarreadings1')
pset.renameArguments(ARG15='lidarreadings2')
pset.renameArguments(ARG16='lidarreadings3')
pset.renameArguments(ARG17='lidarreadings4')
pset.renameArguments(ARG18='lidarreadings5')
pset.renameArguments(ARG19='lidarreadings6')
pset.renameArguments(ARG20='lidarreadings7')
pset.renameArguments(ARG21='lidarreadings8')
pset.renameArguments(ARG22='lidarreadings9')
pset.renameArguments(ARG23='lidarreadings10')
'''
pset.addEphemeralConstant("theta2", lambda:ARG0*2)
pset.addEphemeralConstant("theta3", lambda:ARG0*3)
pset.addEphemeralConstant("theta4", lambda:ARG0*4)
'''
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def EV(individual):
        observation = ENV.reset()
        episode_reward=0
        for step in range(MAX_STEPS):
            action = get_action(observation,individual)
            observation_next,reward,done,info_not = ENV.step(action)
            if done:
                break

            observation = observation_next
            episode_reward += reward
            #if Myclass.count>=2700:
            #    env = wrappers.Monitor(env,'./movie/cartpole-experiment-1')
        return episode_reward,

toolbox.register("evaluate", EV)
toolbox.register("mate", gp.cxOnePoint)
#toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb = 0.1)
toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("expr_mut", gp.genFull, min_= 0, max_ = 2)
#toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut,pset=pset)
toolbox.register("mutate",gp.mutInsert,pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def get_action(observation,individual):
    func = toolbox.compile(expr=individual)

    action =  np.clip(func(observation[0],observation[1],observation[2],observation[3],observation[4],observation[5],observation[6],observation[7],observation[8],observation[9],observation[10],observation[11],observation[12],observation[13],observation[14],observation[15],observation[16],observation[17],observation[18],observation[19],observation[20],observation[21],observation[22],observation[23]),-1,1)
    return action
def varAnd(population,toolbox,cxpb,mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    for i in range(1,len(offspring),2):
        if random.random() < cxpb:
            offspring[i - 1],offspring[i] = toolbox.mate(offspring[i - 1],offspring[i])
            del offspring[i - 1].fitness.values,offspring[i].fitness.values
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring



def main():
        random.seed(1)
        pop = toolbox.population(n=250)
        hof = tools.HallOfFame(1)

        stats_fit= tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)

        mstats = tools.MultiStatistics(fitness=stats_fit,size=stats_size)
        mstats.register("avg", np.mean)
        #mstats.register("std", np.std)
        #mstats.register("min", np.min)
        mstats.register("max", np.max)
        CXPB=0.4
        MUTPB=0.1
        pop,log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats = mstats, halloffame=hof,verbose=True)
        #pop,log = simple(pop, toolbox, CXPB, MUTPB, NGEN, stats = mstats, halloffame=hof,verbose=True)
        return pop, log, hof

if __name__=='__main__':
    pop,log,hof = main()
    print(hof)
    print()
    print()
    print()
    expr = tools.selBest(pop,1)[0]
    print(expr)
    nodes, edges, labels = gp.graph(expr)
    print(nodes)
    print(edges)
    print(labels)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g,prog="dot")
    print(pos)
    nx.draw_networkx_nodes(g,pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g,pos,labels)
    plt.show()

"""
                    fitness             size   
            ----------------------- -----------
gen nevals  avg         max     avg     max
0   250     -22.2845    8.39502 6.844   13 
1   132     -12.9297    8.39502 6.728   13 
2   109     -8.91739    8.39502 7.224   15 
3   108     -7.14805    8.97604 7.304   13 
4   102     -2.24009    8.97604 7.516   15 
5   98      -0.605064   9.9244  7.14    14 
bipedalwalker_gp.py:18: RuntimeWarning: overflow encountered in exp
  return 1 / (1 + np.exp(num))
6   112     1.42744     13.1128 7.12    13 
7   119     3.39087     14.4005 7.356   14 
8   142     2.94838     14.4005 7.008   13 
9   108     3.8613      14.4005 6.7     14 
10  116     3.86475     14.4005 6.4     14 
11  126     2.58688     14.8422 6.168   15 
12  129     1.89083     14.8422 6.248   14 
13  113     3.09872     14.8422 6.156   13 
14  106     4.75753     14.8422 6.116   11 
15  118     2.65761     14.8422 6.304   10 
16  103     2.84603     14.8422 6.396   11 
17  117     3.0099      14.8422 6.4     10 
18  118     2.08339     14.8422 6.128   10 
19  118     2.1419      14.8422 5.904   9  
20  121     1.17056     14.8422 5.808   10 
21  116     0.818383    19.5167 5.54    11 
22  109     2.21796     19.5167 5.316   9  
23  125     0.810582    19.5167 5.26    9  
24  131     1.84517     16.5093 5.204   7  
25  104     4.95283     14.8422 5.156   9  
26  124     1.30968     14.8422 5.144   7  
27  105     -2.30288    14.8422 5.096   7  
28  110     -0.0364109  14.8422 5.124   7  
29  124     -2.11235    14.8422 5.144   7  
30  111     -0.806017   14.8422 5.22    9  
31  120     -0.663452   14.8422 5.156   7  
32  124     -0.530439   14.8422 5.1     7  
33  107     0.63021     14.9263 5.124   7  
34  125     0.0219188   15.0786 5.14    7  
35  116     1.76678     30.163  5.236   8  
36  134     1.60352     30.163  5.26    9  
37  108     3.55304     30.163  5.392   9  
38  124     3.36572     30.163  5.508   10 
39  129     3.49689     30.163  5.404   9  
40  118     1.50547     30.163  5.5     9  
41  129     -0.447877   30.163  5.58    10 
42  130     -0.047666   30.163  5.536   8  
43  113     3.98073     30.163  5.616   9  
44  115     4.45923     30.163  5.704   10 
45  116     5.00223     30.163  5.568   10 
46  109     7.4716      30.163  5.44    10 
47  130     4.76899     30.163  5.376   9  
48  112     5.92366     30.163  5.252   8  
49  119     7.59749     30.163  5.284   8  
50  114     9.63363     30.163  5.268   8  
51  120     8.57035     30.163  5.308   8  
52  120     7.3357      30.163  5.324   10 
53  112     10.9108     30.163  5.22    8  
54  107     7.93452     30.163  5.192   9  
55  125     3.42109     30.163  5.156   8  
56  107     7.25365     30.163  5.144   8  
57  119     2.92688     30.163  5.124   7  
58  104     6.36346     30.163  5.124   7  
59  99      8.36711     30.163  5.132   8  
60  111     6.72147     30.163  5.172   8  
61  125     6.4469      30.163  5.192   8  
62  109     8.35785     30.163  5.132   7  
63  111     8.83666     30.163  5.136   7  
64  115     7.99455     30.163  5.18    7  
65  114     9.05839     30.163  5.152   7  
66  106     11.5251     30.163  5.184   7  
67  135     8.76269     30.163  5.18    7  
68  120     8.94734     30.2158 5.224   8  
69  120     3.5617      30.2158 5.192   9  
70  105     5.62537     30.2158 5.168   11 
71  119     4.12595     30.2158 5.188   10 
72  117     6.57138     30.2158 5.168   7  
73  107     6.64523     30.2158 5.148   7  
74  119     6.55493     30.2158 5.192   7  
75  94      9.9955      30.2158 5.156   9  
76  118     7.00573     30.2158 5.132   9  
77  102     8.00317     30.2158 5.188   9  
78  124     5.74949     30.2158 5.156   9  
79  113     8.85484     30.2158 5.156   9  
80  107     11.2682     30.2158 5.108   7  
81  115     7.26682     30.2158 5.088   7  
82  134     5.78919     30.2158 5.112   7  
83  123     2.87055     30.2158 5.172   8  
84  99      5.92845     30.2158 5.156   7  
85  109     4.32324     30.2158 5.156   7  
86  122     3.60318     30.2158 5.164   7  
87  107     5.86108     30.2158 5.148   7  
88  118     3.40657     30.2158 5.148   7  
89  90      8.41972     30.2158 5.192   9  
90  113     6.49175     30.2158 5.164   7  
91  105     9.79784     30.2158 5.14    8  
92  129     4.39786     30.2158 5.156   9  
93  99      9.47004     30.2158 5.12    7  
94  97      11.2414     30.2158 5.14    8  
95  119     6.06838     30.2158 5.14    7  
96  119     6.73305     30.2158 5.136   7  
97  119     7.01178     30.2158 5.164   7  
98  104     6.5063      30.2158 5.176   9  
99  107     4.48331     30.2158 5.156   7  
100 116     3.74624     30.2158 5.156   7  
101 108     4.14765     30.2158 5.144   7  
102 118     3.54438     30.2158 5.18    7  
103 85      8.11189     30.2158 5.148   7  
104 144     2.30629     30.2158 5.188   8  
105 127     3.14782     30.2158 5.18    9  
106 110     4.79884     30.2158 5.132   8  
107 118     4.59945     30.2158 5.192   8  
108 104     7.84699     30.2158 5.224   8  
109 101     10.3462     37.7474 5.152   8  
110 112     9.09906     37.7474 5.172   8  
111 108     9.32008     37.7474 5.156   9  
112 127     4.73671     37.7474 5.3     11 
113 113     4.1079      37.7474 5.476   9  
114 115     5.4862      37.7474 5.448   10 
115 115     4.08779     37.7474 5.596   11 
116 118     2.93842     37.7474 5.84    10 
117 107     8.28932     37.7474 6.224   11 
118 98      13.6146     49.4137 6.656   11 
119 110     11.1109     49.4137 7.02    12 
120 107     11.1362     49.4137 7.516   12 
121 126     7.56065     49.4137 7.784   14 
122 110     8.55815     49.4137 7.976   13 
123 121     10.4557     49.4137 8.3     14 
124 111     13.9087     52.4174 8.276   13 
125 95      15.5173     92.2983 8.416   13 
126 95      18.0356     52.4174 8.604   13 
127 121     12.4719     52.4174 8.852   15 
128 101     16.2943     52.4174 8.972   16 
129 104     17.5622     52.4174 9.296   15 
130 116     15.3863     52.4174 9.508   17 
131 115     12.7327     52.5165 9.632   17 
132 115     18.1073     52.4174 9.78    20 
133 115     16.5786     68.4723 9.844   21 
134 118     17.7686     68.4723 10      15 
135 114     19.6046     68.4723 10.004  19 
136 121     16.2462     68.4723 10.14   16 
137 101     19.9659     74.1258 10.264  16 
138 137     12.8454     74.1258 10.1    18 
139 110     17.8884     74.1258 9.984   19 
140 101     25.2028     75.4901 9.828   16 
141 98      28.2961     75.4901 9.64    18 
142 122     22.9022     75.4901 9.452   14 
143 129     21.083      75.4901 9.592   17 
144 134     19.3284     75.4901 9.616   17 
145 118     21.2454     75.4901 9.884   17 
146 111     27.0498     75.4901 10.22   18 
147 125     22.253      75.4901 10.36   20 
148 102     32.7753     75.4901 10.424  19 
149 112     30.2568     75.4901 10.496  21 
150 115     27.5038     75.4901 10.308  18 
151 115     26.7176     75.4901 10.436  19 
152 108     28.2723     75.4901 10.452  20 
153 135     24.067      75.4901 10.364  20 
154 121     24.7029     75.4901 10.436  20 
155 122     16.6789     75.4901 10.396  19 
156 96      30.4412     75.4901 10.32   16 
157 112     30.4686     75.4901 10.276  17 
158 124     20.9458     75.4901 10.196  18 
159 117     25.2657     75.4901 10.052  17 
160 113     26.2385     75.4901 10.128  18 
161 123     24.9955     75.4901 10.252  23 
162 111     29.3269     75.4901 10.228  16 
163 120     27.2591     75.4901 10.232  20 
164 113     28.6738     75.4901 10.336  16 
165 105     29.4936     75.4901 10.348  22 
166 107     31.8228     75.4901 10.296  22 
167 115     32.3129     75.4901 10.16   17 
168 108     32.2289     75.4901 10.104  15 
169 119     28.7549     75.4901 10.176  16 
170 107     31.3996     75.4901 10.176  17 
171 121     22.3154     75.4901 10.136  17 
172 115     25.6627     75.4901 10.268  22 
173 131     21.6519     75.4901 10.168  16 
174 117     24.2923     75.4901 10.224  19 
175 125     20.5528     75.4901 10.296  17 
176 113     24.6594     75.4901 10.224  16 
177 110     30.4864     75.4901 10.364  17 
178 118     26.2961     75.4901 10.2    20 
179 122     25.7276     75.4901 10.356  18 
180 93      36.9274     75.4901 10.276  17 
181 114     30.9113     75.4901 10.136  15 
182 136     24.198      75.4901 10.208  20 
183 103     30.049      75.4901 10.408  21 
184 123     27.1675     75.4901 10.336  19 
185 123     22.8527     75.4901 10.224  16 
186 119     24.3737     75.4901 10.164  18 
187 109     26.9564     75.4901 10.132  19 
188 116     26.9947     75.4901 10.284  21 
189 122     28.109      75.4901 10.24   18 
190 109     24.0357     75.4901 10.304  17 
191 113     24.687      75.4901 10.384  17 
192 125     19.9447     75.4901 10.324  16 
193 116     24.9287     75.4901 10.34   17 
194 111     23.6683     75.4901 10.404  18 
195 105     30.5109     75.4901 10.356  18 
196 114     30.6708     75.4901 10.3    19 
197 112     34.5323     75.4901 10.4    18 
198 115     30.7035     75.4901 10.292  20 
199 109     31.6272     75.4901 10.28   18 
200 114     28.3974     75.4901 10.264  18 
201 109     30.1154     75.4901 10.296  18 
202 120     27.9768     75.4901 10.36   18 
203 126     24.5214     75.4901 10.132  18 
204 124     24.2279     79.8701 10.212  20 
205 123     24.1695     79.8701 10.428  20 
206 114     28.0908     79.8701 10.328  19 
207 90      37.7933     79.8701 10.38   20 
208 119     31.7955     79.8701 10.472  19 
209 111     31.7002     79.8701 10.388  20 
210 121     27.2421     79.8701 10.648  18 
211 114     28.0263     79.8701 10.824  20 
212 117     25.803      79.8701 10.94   24 
213 103     31.8466     79.8701 11.276  24 
214 117     29.1418     79.8701 11.796  20 
215 120     27.4399     79.8701 12.224  22 
216 127     25.7734     79.8701 12.488  25 
217 124     23.129      79.8701 12.98   25 
218 113     27.5793     79.8701 13.088  28 
219 113     28.9284     79.8701 13.28   23 
220 114     30.3522     79.8701 13.184  22 
221 121     29.0786     79.8701 13.404  26 
222 112     30.8958     79.8701 13.384  26 
223 112     31.5587     79.8701 13.332  27 
224 105     38.2431     79.8701 13.32   22 
225 122     35.074      82.7055 13.348  25 
226 108     36.3312     82.7055 13.548  31 
227 114     33.258      82.7055 13.24   21 
228 134     22.6385     82.7055 13.02   22 
229 115     27.718      79.8701 13.128  25 
230 100     32.697      79.8701 13.26   26 
231 119     28.7009     79.8701 13.328  30 
232 104     31.282      79.8701 13.3    28 
233 121     26.7772     82.3423 13.616  28 
234 114     33.2322     82.3423 13.364  22 
235 109     28.8116     82.3423 13.564  26 
236 111     30.3485     82.3423 13.556  26 
237 107     34.0416     82.3423 13.556  26 
238 112     35.6946     82.3423 13.596  26 
239 123     29.454      98.5677 13.872  27 
240 113     33.1065     82.3423 14.288  29 
241 123     31.2119     82.3423 14.516  27 
242 114     26.4622     82.3423 14.892  27 
243 105     29.7926     82.3423 15.256  28 
244 106     33.3102     82.3423 15.624  26 
245 103     35.6981     82.3423 16.108  27 
246 135     26.6807     82.3423 16.152  37 
247 116     31.0646     82.3423 16.204  37 
248 140     21.5133     82.3423 16.612  37 
249 117     25.9349     82.3423 16.672  37 
250 125     28.578      82.3423 16.212  35 
251 126     26.4182     82.3423 16.464  35 
252 122     32.642      82.3423 16.404  37 
253 124     27.3291     82.3423 16.628  37 
254 109     33.0026     82.3423 16.412  31 
255 105     33.3413     82.3423 16.168  27 
256 121     28.0749     82.3423 16.12   29 
257 113     27.8179     82.3423 16.268  39 
258 116     29.2195     82.3423 16.308  35 
259 121     27.7344     82.3423 16.108  31 
260 122     26.9517     82.3423 15.888  37 
261 106     31.3426     82.3423 16.224  31 
262 120     28.9671     82.3423 16.16   28 
263 116     28.5585     82.3423 16.232  32 
264 110     32.6124     82.3423 16.404  40 
265 112     33.6334     82.3423 16.46   37 
266 134     28.0357     85.8634 16.028  32 
267 101     37.8312     82.3423 16.148  31 
268 121     32.3071     82.3423 16.168  29 
269 106     34.7239     82.3423 16.208  29 
270 105     35.6185     82.3423 16.064  29 
271 113     30.7044     82.3423 16.06   33 
272 114     30.5045     82.3423 16.156  27 
273 100     33.7815     82.3423 16.144  36 
274 108     33.8984     82.3423 16.04   27 
275 104     36.14       95.7876 16.244  34 
276 116     29.3228     82.3423 16.028  29 
277 124     29.8624     82.3423 16.312  32 
278 96      38.9816     82.3423 16.28   34 
279 120     32.7661     82.3423 16.192  27 
280 115     31.5844     82.3423 16.472  27 
281 115     33.5294     82.3423 16.304  31 
282 115     35.0836     82.3423 16.288  31 
283 112     35.8645     82.3423 16.624  32 
284 121     31.4532     82.3423 16.376  36 
285 135     21.8321     82.3423 16.584  38 
286 124     22.9012     82.3423 16.596  38 
287 100     30.5157     82.3423 16.688  38 
288 91      37.8717     82.3423 16.12   31 
289 106     34.4392     82.3423 15.968  33 
290 125     29.9703     82.3423 16.248  43 
291 117     29.2502     82.3423 16.216  29 
292 108     32.3506     82.3423 16.248  31 
293 109     32.3963     82.3423 16.176  38 
294 106     34.1545     82.3423 16.132  32 
295 130     31.9676     82.3423 16.272  33 
296 103     36.9389     82.3423 16.008  29 
297 103     35.9625     82.3423 16.2    32 
298 112     34.6766     82.3423 16.28   29 
299 125     27.0179     82.3423 16.368  35 
300 120     25.4975     82.3423 16.336  27 
"""