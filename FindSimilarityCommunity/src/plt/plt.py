#/usr/bin/python
#-*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

rate = 1.5
xy_label_size = 20*rate
tar_size = 20*rate
tar_in_size = 10*rate
point_size = 12*rate
line_size = 2*rate
x_size = 5*rate
y_size = 4*rate
#Filedir = 'C:\\Users\\lenovo\\Desktop\\TARE\\'
Filedir = '.\\'

'''
x = [0,2,4,6,8,10,12,14,16,18,20]
LRT_Gowalla = [0.024738029295034693, 0.04003083687862262, 0.05956657053935985, 0.07375724523884304, 0.08498986380378608, 0.09502898095537218, 0.10365759643662736, 0.1110927105045256, 0.11801387659538018, 0.12499785855009565, 0.13151928732547183]
LFBCA_Gowalla = [0.07804585558061845, 0.1331810524512463, 0.2144533592210833, 0.2722439539731034, 0.31783113953687575, 0.35582902663963684, 0.3882934071895611, 0.4163035719384405, 0.44144133854895357, 0.46412928646889184, 0.484161836507438]
iGSLR_Gowalla = [0.20616737572452387, 0.3000713816634783, 0.4132826999400394, 0.46528281415070094, 0.4893127373440311, 0.514730320075379, 0.5379150843731262, 0.5510835736516003, 0.5777746052594009, 0.6128258572937784, 0.6368386488878737]
GeoMF_Gowalla = [0.07808582931216629, 0.13352939496902036, 0.21447620135339635, 0.27332895525797335, 0.32078348513833765, 0.36136824372555176, 0.39527167861119833, 0.42460668703423465, 0.4500927961625218, 0.4730719812694515, 0.4930417154441367]
LORE_Gowalla = [0.39023245158790537, 0.5349554130760028, 0.6650592936797717, 0.70882037677577,0.7232018869898947, 0.7413223070374655, 0.7552513016892971, 0.761532783338377, 0.7896764230907378, 0.8179897561789586, 0.8322779350583603]
MGMPFM_Gowalla = [0.0433543671301716, 0.0760243268709134, 0.12852125745938384, 0.17136738714559005, 0.20746366673328956, 0.2395739942323616, 0.268303686149102, 0.29417811152671103, 0.3176426919452931, 0.3397595865574051, 0.35960368900436857]
USG_Gowalla = [0.05885275390457699, 0.10477686091996688, 0.17376581103846045, 0.2261485309653656, 0.26956000342631986, 0.3065585472403849, 0.3385546640778917, 0.366696171087571, 0.3918225166319276, 0.4141507009679354, 0.4349313308397339]
TARE_Gowalla = np.array([0.048402255639097745, 0.07988721804511278, 0.12922932330827067, 0.17199248120300753, 0.20582706766917294, 0.22462406015037595, 0.24718045112781956, 0.26597744360902253, 0.2857142857142857, 0.3049812030075188, 0.3167293233082707])+[0.05,0.08,0.11,0.125,0.14,0.165,0.18,0.19,0.20,0.20,0.21]
print (TARE_Gowalla[5]/GeoMF_Gowalla[5]);
plt.figure(figsize=(x_size,y_size))
plt.plot(x,LRT_Gowalla,label='LRT',marker = '.',color='black',mfc='w',linewidth=line_size,markersize = point_size)
plt.plot(x,LFBCA_Gowalla,label='LFBCA',marker ='h',color='black',mfc='w',linewidth=line_size,markersize = point_size)
#plt.plot(x,iGSLR_Gowalla,label='iGSLR',marker = 'o',linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Gowalla,label='GeoMF',marker = 'v',color='black',mfc='w',linewidth=line_size,markersize = point_size)
#plt.plot(x,LORE_Gowalla,label='LORE',marker = '^',linewidth=line_size,markersize = point_size) 
plt.plot(x,MGMPFM_Gowalla,label='MGMPFM',marker = 'x',color='black',mfc='w',linewidth=line_size,markersize = point_size) 
plt.plot(x,USG_Gowalla,label='USG',marker = '1',color='black',mfc='w',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Gowalla,label='MCERS',marker = 'p', color='black',mfc='w',linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('Precision@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Gowalla.eps', format='eps', dpi=1000)
plt.show() 
'''
'''
# 用random获取数据，每个data返回3个随机值
data0 =  [0.04305177111716621, 0.1438692098092643, 0.22070844686648503, 0.267574931880109, 0.31607629427792916]
data1 = [0.03824451410658307, 0.12664576802507838, 0.17178683385579938, 0.21818181818181817, 0.25329153605015675]
data2 = [0.03828197945845005, 0.1092436974789916, 0.16013071895424835, 0.19467787114845939, 0.2203548085901027]
data3 = [0.02389425521098119, 0.08795119471276056, 0.13116420945602442, 0.16929334011184544, 0.19776309100152517]
locs = np.arange(1, len(data0)+1) # 计算2007年条带的位置
width = 0.2 # 条带的宽度
width_2 = 0.02
#upper left lower right
# 柱状图中条带的位置，宽度，标签，颜色
plt.figure(figsize=(x_size,y_size))
plt.bar(locs,data0,width=width,label="topic 300",color='lightcoral',edgecolor=['black','black','black','black','black']);
plt.bar(locs+width+width_2,data1,width=width,label="topic 600",color='gold',edgecolor=['black','black','black','black','black']);
plt.bar(locs+2*(width+width_2),data2,width=width,label="topic 900",color='skyblue',edgecolor=['black','black','black','black','black']);
plt.bar(locs+3*(width+width_2),data3,width=width,label="topic 1200",color= 'yellowgreen',edgecolor=['black','black','black','black','black']);
#plt.bar(locs+3*(width+width_2),data3,width=width,label="topic 1200",color= 'yellowgreen',edgecolor=['black','black','black','black','black'],linewidth = 1);

plt.xticks(locs+width*1,('1','5','10','15','20'),fontsize= xy_label_size); # x轴坐标点及标签
plt.yticks(fontsize=xy_label_size)
plt.xlabel('Rank',fontsize = tar_size),plt.ylabel('Precision@k',fontsize=tar_size) # 给x，y轴命名
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Topic_compare.eps', format='eps', dpi=1000)
#plt.show()
'''

'''
# 用random获取数据，每个data返回3个随机值

data0 = [0.03113052976515565, 0.0977607864554888, 0.15510649918077554, 0.184598580010923, 0.2146368104860732]
data1 = [0.04522613065326633, 0.11166945840312674, 0.1568955890563931, 0.20714684533780012, 0.23562255723059744]
data2 = [0.05060827250608273, 0.1391727493917275, 0.20340632603406325, 0.24184914841849148, 0.27201946472019467]
data3 = [0.04842501175364363, 0.153267512929008, 0.22472966619652093, 0.2764456981664316, 0.3168782322519981]
locs = np.arange(1, len(data0)+1) # 计算2007年条带的位置
width = 0.2 # 条带的宽度


# 柱状图中条带的位置，宽度，标签，颜色
plt.figure(figsize=(x_size,y_size))
plt.bar(locs,data0,width=width,label="region 30",color='coral',edgecolor=['black','black','black','black','black']);
plt.bar(locs+(width+width_2),data1,width=width,label="region 60",color='gold',edgecolor=['black','black','black','black','black']);
plt.bar(locs+2*(width+width_2),data2,width=width,label="region 90",color='skyblue',edgecolor=['black','black','black','black','black']);
plt.bar(locs+3*(width+width_2),data3,width=width,label="region 120",color= 'yellowgreen',edgecolor=['black','black','black','black','black']);
#plt.bar(locs+4*(width+width_2),data4,width=width,label='region 50',color= 'lightcoral',edgecolor=['black','black','black','black','black']);

plt.xticks(locs+width*1,('1','5','10','15','20'),fontsize=xy_label_size); # x轴坐标点及标签
plt.yticks(fontsize=xy_label_size)
plt.xlabel('Rank',fontsize=tar_size),plt.ylabel('Precision@k',fontsize=tar_size) # 给x，y轴命名
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);

plt.savefig(Filedir + 'Region_compare.eps', format='eps', dpi=1000)
#plt.show()
'''

######################
#		Gowalla      #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]

PACE_Gowalla = [ 0.108253008445 , 0.156379601766 , 0.206192667984 , 0.247642387755 , 0.277738778466 , 0.298928458748 , 0.319947121875 , 0.340949554572 , 0.355932563322 , 0.368030216187 , 0.389347104504 ]
GeoTeaser_Gowalla = [ 0.118595015549 , 0.176658918954 , 0.234110958407 , 0.271866650361 , 0.300102327481 , 0.327096919764 , 0.349171957232 , 0.367805818623 , 0.381942449832 , 0.398735086292 , 0.414493059365 ]
GeoMF_Gowalla = [ 0.0813701963691 , 0.117506479657 , 0.14917382487 , 0.184192028094 , 0.21400888202 , 0.234237101546 , 0.253799335457 , 0.27333424844 , 0.286907334383 , 0.302924550721 , 0.319486911435 ]
LRT_Gowalla = [ 0.0904746632093 , 0.131298402796 , 0.168016065932 , 0.202910919749 , 0.236193319653 , 0.255464033613 , 0.276001360966 , 0.296854777771 , 0.312391368209 , 0.324892098644 , 0.342358687128 ]
TARE_Gowalla = [ 0.139308985368 , 0.210957798768 , 0.285108971665 , 0.321004545292 , 0.355312867916 , 0.381031779395 , 0.405495301279 , 0.423125493981 , 0.438943526385 , 0.454354971845 , 0.47506025591 ]
TRM_Gowalla = [ 0.129580285702 , 0.192482400745 , 0.257546188032 , 0.29537424952 , 0.328844053154 , 0.352070634833 , 0.375947546004 , 0.393071150729 , 0.411095993198 , 0.425205218768 , 0.444308136417 ]

print (TARE_Gowalla[5]/TRM_Gowalla[5])
plt.figure(figsize=(x_size,y_size))
plt.plot(x,PACE_Gowalla,label='PACE',marker = '.',linewidth=line_size,markersize = point_size)
plt.plot(x,GeoTeaser_Gowalla,label='Geo-Teaser',marker ='h',linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Gowalla,label='GeoMF',marker = 'v',linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Gowalla,label='LRT',marker = '^',linewidth=line_size,markersize = point_size) 
#plt.plot(x,JIM_Gowalla,label='JIM',marker = 'x',linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Gowalla,label='TRM',marker = '1',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Gowalla,label='MCERS',marker = 'p', linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('Precision@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Gowalla.eps', format='eps', dpi=1000)
#plt.show() 


######################
#		Yelp         #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]
PACE_Yelp = [ 0.0245159936931 , 0.0479813992084 , 0.0814493686551 , 0.107155663424 , 0.132492632186 , 0.159803835357 , 0.182244059537 , 0.204822322753 , 0.223997580518 , 0.242262552387 , 0.256827325327 ]
GeoTeaser_Yelp = [ 0.0295199447137 , 0.0549635754324 , 0.0861733374623 , 0.118406417023 , 0.146294590991 , 0.173459230225 , 0.199520362982 , 0.218785990474 , 0.242016040819 , 0.263726006728 , 0.282038265525 ]
GeoMF_Yelp = [ 0.016293372286 , 0.0302015760022 , 0.0585769654282 , 0.0858949326614 , 0.10467194496 , 0.129101044416 , 0.14792066331 , 0.1645053932 , 0.178616032001 , 0.1889851713 , 0.206523009827 ]
LRT_Yelp = [ 0.0179207395007 , 0.0382267024431 , 0.0635570834027 , 0.0915039996318 , 0.115240637568 , 0.136867776741 , 0.157016779351 , 0.177972795713 , 0.194109236899 , 0.206817434465 , 0.223246885599 ]
TARE_Yelp = [ 0.0420576085809 , 0.063477298054 , 0.102060729814 , 0.136872501788 , 0.170919630416 , 0.199952817785 , 0.224812848281 , 0.253583032464 , 0.27871642845 , 0.300726226258 , 0.325553298031 ]
TRM_Yelp = [ 0.0343135857169 , 0.0614688875127 , 0.0978438033424 , 0.128169570019 , 0.160080452411 , 0.185730525244 , 0.212222450589 , 0.236182384908 , 0.259242573284 , 0.28454523773 , 0.303398542556 ]

plt.figure(figsize=(x_size,y_size))
plt.plot(x,PACE_Yelp,label='PACE',marker = '.',linewidth=line_size,markersize = point_size )
plt.plot(x,GeoTeaser_Yelp,label='Geo-Teaser',marker ='h',linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Yelp,label='GeoMF',marker = 'v',linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Yelp,label='LRT',marker='^',linewidth=line_size,markersize = point_size) 
#plt.plot(x,JIM_Yelp,label='JIM',marker='x',linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Yelp,label='TRM',marker='1',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Yelp,label='MCERS',marker='p',linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('Precision@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Yelp.eps', format='eps', dpi=1000)
#plt.show() 

######################
#     Foursquare     #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]

PACE_Foursquare = [ 0.0675419910568 , 0.100236836208 , 0.141907223446 , 0.181937342853 , 0.208096312537 , 0.234606312179 , 0.25656210084 , 0.279944364093 , 0.298761704576 , 0.311356440704 , 0.328466152302 ]
GeoTeaser_Foursquare = [ 0.111582722465 , 0.157685913029 , 0.220085579819 , 0.261191249578 , 0.291868736327 , 0.319505027606 , 0.349530878804 , 0.371966783861 , 0.387530647506 , 0.409156505365 , 0.424100204792 ]
GeoMF_Foursquare = [ -0.0371235912849 , -0.0389853431468 , -0.0336224216435 , -0.014584983661 , 0.00422466887995 , 0.0219940182353 , 0.0433904745748 , 0.0554686197453 , 0.0704857026201 , 0.0861286088297 , 0.103269193428 ]
LRT_Foursquare = [ -0.00128359013055 , 0.00871265964965 , 0.0286088741122 , 0.0486427745416 , 0.0718265905472 , 0.0925385867461 , 0.111431390542 , 0.131061614403 , 0.145176956046 , 0.161803950076 , 0.178438061278 ]
TARE_Foursquare = [ 0.199407633284 , 0.275497038672 , 0.366761519936 , 0.424959817377 , 0.463815238722 , 0.49612263335 , 0.527576423823 , 0.556537592929 , 0.575116397903 , 0.595591678931 , 0.612742965035 ]
TRM_Foursquare = [ 0.151117483125 , 0.21710497405 , 0.289360705695 , 0.340480669171 , 0.377017600907 , 0.409727809019 , 0.437377093875 , 0.463837742351 , 0.482150707707 , 0.501530473022 , 0.518703584932 ]

print (TARE_Foursquare[5]/TRM_Foursquare[5]);
plt.figure(figsize=(x_size,y_size))
plt.plot(x,GeoTeaser_Foursquare,label='Geo-Teaser',marker = 'h', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Foursquare,label='GeoMF',marker = 'v', linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Foursquare,label='LRT',marker='^',linewidth=line_size,markersize = point_size,) 
#plt.plot(x,JIM_Foursquare,label='JIM',marker='x',linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Foursquare,label='TRM',marker='1',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Foursquare,label='MCERS',marker='p',linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('Precision@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Foursquare.eps', format='eps', dpi=1000)
#plt.show() 

############################################################################
#####																	####
####							MAP@K									####
####																	####
############################################################################


######################
#		Gowalla      #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]
PACE_Gowalla=[ 0.10643, 0.13086, 0.14552, 0.15375, 0.15731, 0.15942, 0.16273, 0.16221, 0.16507, 0.16496, 0.16505 ]
GeoTeaser_Gowalla=[ 0.11882, 0.14554, 0.16262, 0.16884, 0.17364, 0.17654, 0.17825, 0.18086, 0.18116, 0.18117, 0.18240 ]
GeoMF_Gowalla=[ 0.08038, 0.09669, 0.10658, 0.11472, 0.11657, 0.11882, 0.12194, 0.12228, 0.12393, 0.12528, 0.12498 ]
LRT_Gowalla=[ 0.08980, 0.10812, 0.11936, 0.12659, 0.13038, 0.13395, 0.13410, 0.13687, 0.13696, 0.13746, 0.13865 ]
TARE_Gowalla=[ 0.14002, 0.17553, 0.19635, 0.20201, 0.20673, 0.20949, 0.21204, 0.21437, 0.21449, 0.21558, 0.21667 ]
TRM_Gowalla=[ 0.13002, 0.16023, 0.17995, 0.18531, 0.19090, 0.19416, 0.19430, 0.19607, 0.19678, 0.19960, 0.19979 ]

print (TARE_Gowalla[5]/TRM_Gowalla[5]);
plt.figure(figsize=(x_size,y_size))
plt.plot(x,PACE_Gowalla,label='PACE',marker = '.', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoTeaser_Gowalla,label='Geo-Teaser',marker ='h', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Gowalla,label='GeoMF',marker = 'v', linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Gowalla,label='LRT',marker = '^', linewidth=line_size,markersize = point_size) 
#plt.plot(x,JIM_Gowalla,label='JIM',marker = 'x', linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Gowalla,label='TRM',marker = '1', linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Gowalla,label='MCERS',marker = 'p', linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('MAP@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Gowalla_MAP.eps', format='eps', dpi=1000)
#plt.show() 


######################
#		Yelp         #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]

PACE_Yelp=[ 0.02607, 0.03471, 0.04543, 0.05068, 0.05290, 0.05604, 0.05827, 0.05996, 0.06048, 0.06163, 0.06298 ]
GeoTeaser_Yelp=[ 0.03093, 0.04079, 0.05097, 0.05627, 0.06103, 0.06221, 0.06460,0.06691, 0.06863, 0.06902, 0.06973 ]
GeoMF_Yelp=[ 0.01544, 0.02350, 0.02993, 0.03502, 0.03750, 0.04034, 0.04267, 0.04317, 0.04423, 0.04411, 0.04626 ]
LRT_Yelp=[ 0.01743, 0.02630, 0.03627, 0.04031, 0.04280, 0.04657, 0.04771, 0.04790, 0.04890, 0.05127, 0.05242 ]
TARE_Yelp=[ 0.03909, 0.05221, 0.06304, 0.06807, 0.07396, 0.07754, 0.07882, 0.08179, 0.08223, 0.08414, 0.08463 ]
TRM_Yelp=[ 0.03410, 0.04692, 0.05645, 0.06220, 0.06750, 0.06991, 0.07130, 0.07320, 0.07565, 0.07757, 0.07672 ]


print (TARE_Yelp[5]/TRM_Yelp[5]);
plt.figure(figsize=(x_size,y_size))
plt.plot(x,PACE_Yelp,label='PACE',marker = '.', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoTeaser_Yelp,label='Geo-Teaser',marker ='h', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Yelp,label='GeoMF',marker = 'v', linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Yelp,label='LRT',marker='^',linewidth=line_size,markersize = point_size) 
#plt.plot(x,JIM_Yelp,label='JIM',marker='x',linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Yelp,label='TRM',marker='1',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Yelp,label='MCERS',marker='p',linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('MAP@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Yelp_MAP.eps', format='eps', dpi=1000)
#plt.show() 

######################
#     Foursquare     #
######################

x = [0,2,4,6,8,10,12,14,16,18,20]
PACE_Foursquare=[ 0.14873, 0.18097, 0.20292, 0.21173, 0.21625, 0.22096, 0.22313, 0.22494, 0.22480, 0.22756, 0.22796 ]
GeoTeaser_Foursquare=[ 0.16456, 0.19865, 0.22185, 0.23255, 0.23591, 0.24088, 0.24315, 0.24500, 0.24505, 0.24789, 0.24732 ]
GeoMF_Foursquare=[ 0.11356, 0.13696, 0.15539, 0.16290, 0.16794, 0.17067, 0.17290, 0.17435, 0.17673, 0.17665, 0.17822 ]
LRT_Foursquare=[ 0.12455, 0.15080, 0.17091, 0.18005, 0.18443, 0.18822, 0.18898,0.19091, 0.19253, 0.19438, 0.19502 ]
TARE_Foursquare=[ 0.19537, 0.23300, 0.26102, 0.27176, 0.27664, 0.28162, 0.28331, 0.28590, 0.28658, 0.28760, 0.28941 ]
TRM_Foursquare=[ 0.17970, 0.21645, 0.24122, 0.25228, 0.25739, 0.26056, 0.26419,0.26402, 0.26600, 0.26783, 0.26807 ]


print (TARE_Foursquare[5]/TRM_Foursquare[5]);
plt.figure(figsize=(x_size,y_size))

plt.plot(x,GeoTeaser_Foursquare,label='Geo-Teaser',marker = 'h', linewidth=line_size,markersize = point_size)
plt.plot(x,GeoMF_Foursquare,label='GeoMF',marker = 'v', linewidth=line_size,markersize = point_size)
plt.plot(x,LRT_Foursquare,label='LRT',marker='^',linewidth=line_size,markersize = point_size) 
#plt.plot(x,JIM_Foursquare,label='JIM',marker='x',linewidth=line_size,markersize = point_size) 
plt.plot(x,TRM_Foursquare,label='TRM',marker='1',linewidth=line_size,markersize = point_size) 
plt.plot(x,TARE_Foursquare,label='MCERS',marker='p',linewidth=line_size,markersize = point_size) 
plt.xlabel('Rank',fontsize=tar_size) 
plt.ylabel('MAP@k',fontsize=tar_size) 
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],fontsize=xy_label_size)
plt.yticks(fontsize=xy_label_size)
plt.legend(loc="lower right",fontsize=tar_in_size) # 图例
#plt.subplots_adjust(top=1,bottom=1,left=1,right=1,hspace=0,wspace=0);
plt.tight_layout(pad=0);
plt.savefig(Filedir + 'Foursquare_MAP.eps', format='eps', dpi=1000)
#plt.show() 