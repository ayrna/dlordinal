import os
import shutil
import tarfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import PIL.Image as Image
import pytest
import torch
from torchvision.transforms import ToTensor

from dlordinal.datasets import Adience
from dlordinal.datasets.adience import _image_path_from_row, _track_progress


def _create_adience_data(tmp_path):
    tmp_path = Path(tmp_path) / "adience"
    tmp_path.mkdir(parents=True, exist_ok=True)

    images_path = tmp_path / "aligned"
    folds_path = tmp_path / "folds"

    images_path.mkdir(parents=True, exist_ok=True)
    folds_path.mkdir(parents=True, exist_ok=True)

    folds = [
        """user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\ttilt_ang\tfiducial_yaw_angle\tfiducial_score
30601258@N03\t10399646885_67c7d20df9_o.jpg\t1\t(25, 32)\tf\t0\t414\t1086\t1383\t-115\t30\t17
30601258@N03\t10424815813_e94629b1ec_o.jpg\t2\t(25, 32)\tm\t301\t105\t640\t641\t0\t0\t94
30601258@N03\t10437979845_5985be4b26_o.jpg\t1\t(25, 32)\tf\t2395\t876\t771\t771\t175\t-30\t74
30601258@N03\t10437979845_5985be4b26_o.jpg\t3\t(25, 32)\tm\t752\t1255\t484\t485\t180\t0\t47
30601258@N03\t11816644924_075c3d8d59_o.jpg\t2\t(25, 32)\tm\t175\t80\t769\t768\t-75\t0\t34
30601258@N03\t11562582716_dbc2eb8002_o.jpg\t1\t(25, 32)\tf\t0\t422\t1332\t1498\t-100\t15\t54
30601258@N03\t10424595844_1009c687e4_o.jpg\t4\t(38, 43)\tf\t1912\t905\t1224\t1224\t155\t0\t64
30601258@N03\t9506931745_796300ca4a_o.jpg\t5\t(25, 32)\tf\t1069\t581\t1575\t1575\t0\t30\t131
30601258@N03\t10190308156_5c748ab2da_o.jpg\t5\t(25, 32)\tf\t474\t1893\t485\t484\t-115\t30\t55
30601258@N03\t10190308156_5c748ab2da_o.jpg\t2\t(25, 32)\tm\t1013\t1039\t453\t452\t-75\t0\t59
30601258@N03\t11624488765_9db0b93c94_o.jpg\t2\t(25, 32)\tm\t101\t56\t740\t740\t-90\t0\t75
30601258@N03\t10204739113_0e2ae11708_o.jpg\t6\t(25, 32)\tm\t336\t640\t841\t842\t-85\t0\t94
30601258@N03\t10204739113_0e2ae11708_o.jpg\t1\t(25, 32)\tf\t693\t247\t720\t720\t-85\t30\t132
30601258@N03\t11518638385_cac7193c86_o.jpg\t2\t(25, 32)\tm\t87\t20\t728\t728\t-95\t0\t79
30601258@N03\t11341941104_2bcd4b99e0_o.jpg\t1\t(25, 32)\tf\t1039\t1432\t624\t625\t185\t30\t120
30601258@N03\t11431644464_5510e0b7e9_o.jpg\t2\t(25, 32)\tm\t223\t58\t780\t781\t-85\t0\t40
30601258@N03\t11562657036_5fe2235bed_o.jpg\t5\t(25, 32)\tf\t518\t234\t444\t444\t-15\t0\t78
30601258@N03\t11438175534_c13ee0375c_o.jpg\t2\t(25, 32)\tm\t890\t229\t746\t746\t-105\t30\t132
30601258@N03\t11438175534_c13ee0375c_o.jpg\t1\t(25, 32)\tf\t996\t1222\t733\t733\t-85\t0\t109
30601258@N03\t10571000386_90e4070c7c_o.jpg\t2\t(25, 32)\tm\t596\t156\t684\t688\t-5\t0\t28
10044155@N06\t11345830753_1574997964_o.jpg\t153\t(48, 53)\tm\t1367\t1502\t242\t243\t180\t0\t98
10044155@N06\t9345643869_4353a29134_o.jpg\t180\t(38, 43)\tf\t1585\t524\t324\t325\t10\t0\t25
10044155@N06\t9345643869_4353a29134_o.jpg\t140\t(38, 43)\tm\t981\t281\t308\t309\t5\t30\t45
10044155@N06\t10745058173_97ba579984_o.jpg\t181\t(38, 43)\tm\t578\t0\t192\t154\t0\t30\t7
10044155@N06\t11345535113_0298e0a9b8_o.jpg\t149\t(60, 100)\tm\t2132\t863\t306\t306\t-5\t30\t44
10044155@N06\t11331359584_70f228b11a_o.jpg\t139\t(38, 43)\tm\t1055\t814\t382\t382\t10\t0\t31
10044155@N06\t11331339346_04d596b4bf_o.jpg\t133\t(38, 43)\tf\t1378\t532\t427\t427\t10\t0\t51
10044155@N06\t11345511995_449a374ae8_o.jpg\t155\t(25, 32)\tm\t899\t405\t449\t448\t15\t45\t39
10044155@N06\t11345511995_449a374ae8_o.jpg\t140\t(38, 43)\tm\t445\t274\t344\t344\t10\t30\t85
10044155@N06\t11345760473_128b13453c_o.jpg\t145\t(38, 43)\tm\t980\t970\t242\t242\t-10\t0\t104
10044155@N06\t11345525826_8798fa00f2_o.jpg\t182\t(38, 43)\tf\t514\t2091\t274\t274\t0\t0\t32
10044155@N06\t9345543387_b7ca38c8be_o.jpg\t140\t(38, 43)\tm\t1113\t379\t260\t260\t10\t0\t49
10044155@N06\t9345543387_b7ca38c8be_o.jpg\t183\t(25, 32)\tf\t1387\t547\t200\t201\t0\t0\t120
10044155@N06\t9345543387_b7ca38c8be_o.jpg\t184\t(48, 53)\tm\t1809\t556\t192\t191\t0\t30\t57
10044155@N06\t11345711336_7a70c81d07_o.jpg\t146\t(48, 53)\tm\t823\t1050\t268\t268\t-15\t0\t127
10044155@N06\t11345711336_7a70c81d07_o.jpg\t132\t(25, 32)\tf\t1031\t1059\t261\t262\t-15\t0\t133
10044155@N06\t11331320526_f8635e254e_o.jpg\t148\t(48, 53)\tf\t398\t854\t242\t242\t-5\t0\t101
10044155@N06\t11331320526_f8635e254e_o.jpg\t134\t(48, 53)\tm\t986\t846\t242\t242\t0\t0\t119
10044155@N06\t11331320526_f8635e254e_o.jpg\t145\t(38, 43)\tm\t1533\t833\t242\t242\t0\t0\t84
10044155@N06\t11331320526_f8635e254e_o.jpg\t149\t(60, 100)\tm\t688\t809\t242\t242\t0\t-15\t112""",
        """user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\ttilt_ang\tfiducial_yaw_angle\tfiducial_score
114841417@N06\t12068804204_085d553238_o.jpg\t481\t(60, 100)\tf\t1141\t780\t975\t976\t0\t0\t118
114841417@N06\t12068804204_085d553238_o.jpg\t482\t(48, 53)\tm\t1821\t283\t969\t969\t-25\t15\t35
114841417@N06\t12078357226_5fdd9367de_o.jpg\t483\t(4, 6)\tf\t1788\t341\t306\t306\t-10\t0\t168
114841417@N06\t12019067874_0e988248af_o.jpg\t483\t(4, 6)\tf\t3\t183\t932\t777\t-115\t0\t27
114841417@N06\t12077009614_2490487d2a_o.jpg\t484\t45\tf\t258\t133\t1734\t1734\t15\t0\t11
114841417@N06\t12060557503_813b9599be_o.jpg\t483\t(4, 6)\tf\t857\t1157\t357\t357\t-90\t-15\t12
114841417@N06\t12059865494_dace7a1325_o.jpg\t485\t13\tf\t1346\t294\t1001\t1001\t10\t0\t102
114841417@N06\t12101458663_c5be3d6a8f_o.jpg\t483\t(4, 6)\tf\t0\t1051\t307\t351\t-110\t-15\t14
114841417@N06\t12061744626_215481e333_o.jpg\t486\t(15, 20)\tf\t1735\t729\t287\t287\t-10\t0\t89
114841417@N06\t12061744626_215481e333_o.jpg\t487\t(15, 20)\tm\t1321\t809\t261\t262\t-10\t30\t101
114841417@N06\t12061744626_215481e333_o.jpg\t483\t(4, 6)\tf\t2033\t852\t242\t242\t-10\t0\t82
114841417@N06\t12059875396_f5c3a70550_o.jpg\t488\t(15, 20)\tf\t344\t1083\t752\t753\t-85\t0\t103
114841417@N06\t12059875396_f5c3a70550_o.jpg\t485\t13\tf\t415\t444\t702\t701\t-95\t30\t160
114841417@N06\t12076779535_2bf0f4afbb_o.jpg\t489\t35\tf\t187\t267\t708\t693\t-95\t-30\t10
114841417@N06\t12060036015_e7c827be8d_o.jpg\t486\t(15, 20)\tf\t210\t231\t544\t545\t-75\t0\t81
114841417@N06\t12060036015_e7c827be8d_o.jpg\t489\t35\tf\t282\t0\t512\t468\t-115\t30\t114
114841417@N06\t12101188123_0c9af893c9_o.jpg\t490\t(8, 12)\tm\t1399\t288\t733\t733\t170\t15\t95
114841417@N06\t12101188123_0c9af893c9_o.jpg\t485\t13\tf\t2027\t772\t669\t670\t280\t15\t172
114841417@N06\t12076982073_3d7cfa797b_o.jpg\t491\t45\tm\t411\t688\t1676\t1677\t-100\t0\t64
114841417@N06\t12076982073_3d7cfa797b_o.jpg\t492\t(15, 20)\tm\t74\t221\t982\t982\t-70\t0\t62
114841417@N06\t12056671804_3a0df8fd74_o.jpg\t489\t35\tf\t207\t68\t1504\t1505\t-90\t30\t17
114841417@N06\t12077182105_f057ab2d06_o.jpg\t491\t45\tm\t248\t573\t810\t810\t5\t0\t78
114841417@N06\t12077182105_f057ab2d06_o.jpg\t485\t13\tf\t1375\t143\t612\t612\t0\t30\t126
114841417@N06\t12077182105_f057ab2d06_o.jpg\t490\t(8, 12)\tm\t2269\t594\t612\t612\t5\t0\t43
114841417@N06\t12101324215_c104676b85_o.jpg\t483\t(4, 6)\tf\t831\t721\t1626\t1626\t-100\t0\t101
114841417@N06\t12019019424_7719bde328_o.jpg\t489\t35\tf\t208\t392\t524\t524\t185\t0\t75
114841417@N06\t12059615054_edf390a633_o.jpg\t485\t13\tf\t2033\t940\t797\t797\t275\t0\t23
114841417@N06\t12059615054_edf390a633_o.jpg\t490\t(8, 12)\tm\t1345\t454\t746\t746\t170\t0\t59
114841417@N06\t12101057403_eda2051e3d_o.jpg\t490\t(8, 12)\tm\t910\t811\t484\t484\t-90\t15\t87
114841417@N06\t12078845563_da5cd4f54c_o.jpg\t485\t13\tf\t1753\t502\t522\t522\t-15\t30\t75
114841417@N06\t12078845563_da5cd4f54c_o.jpg\t490\t(8, 12)\tm\t1407\t704\t395\t395\t0\t45\t45
114841417@N06\t12077897553_a4fe437157_o.jpg\t483\t(4, 6)\tf\t1386\t704\t778\t778\t-5\t0\t148
114841417@N06\t12100706905_55d117a462_o.jpg\t498\t(15, 20)\tm\t1391\t661\t446\t446\t5\t0\t107
114841417@N06\t12100706905_55d117a462_o.jpg\t485\t13\tf\t1024\t836\t415\t415\t10\t0\t73
114841417@N06\t12101712666_46556d9d38_o.jpg\t483\t(4, 6)\tf\t929\t446\t975\t975\t-100\t0\t145
114841417@N06\t12102011736_93b346a1b3_o.jpg\t489\t35\tf\t0\t0\t3264\t2448\t-110\t-15\t37
114841417@N06\t12056412465_a03caf8f65_o.jpg\t498\t(15, 20)\tm\t1795\t443\t368\t368\t5\t0\t60
114841417@N06\t12056412465_a03caf8f65_o.jpg\t502\t(15, 20)\tm\t975\t480\t339\t339\t-10\t0\t100""",
        """user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\ttilt_ang\tfiducial_yaw_angle\tfiducial_score
64504106@N06\t11831304783_488d6c3a6d_o.jpg\t911\t(0, 2)\tm\t438\t914\t605\t606\t-90\t0\t123
64504106@N06\t11849646776_35253e988f_o.jpg\t911\t(0, 2)\tm\t19\t712\t1944\t1736\t-105\t0\t86
64504106@N06\t11848166326_57b03f535e_o.jpg\t911\t(0, 2)\tm\t382\t680\t1785\t1768\t-80\t0\t14
64504106@N06\t11812546385_bb4d020dde_o.jpg\t911\t(0, 2)\tm\t608\t948\t893\t892\t-105\t0\t8
64504106@N06\t11831118625_81dcc72e75_o.jpg\t912\t(38, 43)\tm\t23\t150\t508\t508\t-90\t0\t79
64504106@N06\t11831118625_81dcc72e75_o.jpg\t913\t(25, 32)\tf\t174\t787\t472\t472\t-90\t0\t40
64504106@N06\t11837596415_11e2a216ce_o.jpg\t911\t(0, 2)\tm\t373\t321\t656\t656\t-90\t0\t84
64504106@N06\t11817152085_7debc19e54_o.jpg\t911\t(0, 2)\tm\t396\t750\t867\t867\t-80\t0\t47
64504106@N06\t11839897733_f3b52ec5b9_o.jpg\t911\t(0, 2)\tm\t537\t591\t1077\t1077\t-100\t-15\t36
64504106@N06\t11817384233_8652174462_o.jpg\t911\t(0, 2)\tm\t1423\t895\t688\t689\t-90\t15\t61
64504106@N06\t11831961146_98ddb57177_o.jpg\t911\t(0, 2)\tm\t605\t202\t1237\t1237\t-75\t0\t98
64504106@N06\t11842475906_0eaf471e6e_o.jpg\t911\t(0, 2)\tm\t831\t541\t969\t969\t-95\t0\t111
64504106@N06\t11856914806_a1f54a948b_o.jpg\t911\t(0, 2)\tm\t0\t878\t1166\t1370\t-95\t0\t121
64504106@N06\t11846140823_aec2247390_o.jpg\t911\t(0, 2)\tm\t438\t51\t1454\t1453\t-5\t0\t72
64504106@N06\t11812700823_eca6f360cf_o.jpg\t911\t(0, 2)\tm\t0\t66\t2325\t2382\t-90\t0\t97
64504106@N06\t11831571504_7044a2e454_o.jpg\t911\t(0, 2)\tm\t468\t645\t1097\t1096\t-100\t0\t81
64504106@N06\t11831157325_dd9e1c96f4_o.jpg\t911\t(0, 2)\tm\t1069\t353\t1326\t1326\t-85\t15\t17
64504106@N06\t11813333234_a68667c7d6_o.jpg\t911\t(0, 2)\tm\t297\t0\t1403\t1252\t-5\t0\t57
64504106@N06\t11856510614_25e6d91c91_o.jpg\t911\t(0, 2)\tm\t1166\t1047\t306\t306\t-95\t0\t38
64504106@N06\t11819581886_40f9d393a3_o.jpg\t911\t(0, 2)\tm\t253\t639\t1383\t1383\t-85\t0\t125""",
        """user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\ttilt_ang\tfiducial_yaw_angle\tfiducial_score
113445054@N07\t11763777465_11d01c34ce_o.jpg\t1322\t(25, 32)\tm\t1102\t296\t357\t357\t-15\t0\t59
113445054@N07\t11763777465_11d01c34ce_o.jpg\t1323\t(25, 32)\tf\t1713\t580\t325\t325\t-5\t0\t118
113445054@N07\t11763777465_11d01c34ce_o.jpg\t1324\t(15, 20)\tf\t1437\t664\t306\t306\t5\t0\t109
113445054@N07\t11764005785_f21921aea6_o.jpg\t1325\t(25, 32)\tf\t978\t229\t803\t803\t-20\t-45\t16
113445054@N07\t11763728674_a41d99f71e_o.jpg\t1326\t(25, 32)\tm\t1745\t910\t242\t242\t-10\t0\t55
113445054@N07\t11764019623_8ffb8ff4f5_o.jpg\t1327\t(25, 32)\tf\t1294\t752\t1013\t1013\t-10\t30\t110
113445054@N07\t11764019623_8ffb8ff4f5_o.jpg\t1325\t(25, 32)\tf\t798\t583\t943\t943\t-10\t15\t57
113445054@N07\t11764019623_8ffb8ff4f5_o.jpg\t1328\t(25, 32)\tf\t2632\t1069\t243\t242\t15\t15\t23
113445054@N07\t11763616596_db19dbce85_o.jpg\t1329\t34\tm\t803\t854\t612\t612\t5\t0\t20
113445054@N07\t11763616596_db19dbce85_o.jpg\t1325\t(25, 32)\tf\t1141\t1282\t503\t504\t5\t0\t72
113445054@N07\t11764137866_0a77db9f90_o.jpg\t1330\t(25, 32)\tf\t422\t648\t688\t689\t0\t0\t39
113445054@N07\t11764137866_0a77db9f90_o.jpg\t1331\t(38, 43)\tf\t1168\t466\t573\t574\t-15\t0\t70
113445054@N07\t11763046045_3be94e42a1_o.jpg\t1325\t(25, 32)\tf\t667\t750\t472\t472\t5\t0\t83
113445054@N07\t11763046045_3be94e42a1_o.jpg\t1332\t(25, 32)\tf\t1074\t741\t459\t459\t-10\t0\t73
113445054@N07\t11763511025_786a7a8662_o.jpg\t1325\t(25, 32)\tf\t1114\t945\t924\t924\t0\t30\t15
113445054@N07\t11802734256_1073ecc435_o.jpg\t1333\t(25, 32)\tf\t514\t604\t1090\t1090\t-25\t0\t55
113445054@N07\t11802734256_1073ecc435_o.jpg\t1334\t(25, 32)\tf\t1995\t333\t453\t561\t-10\t-15\t25
113445054@N07\t11763981535_b191b65fda_o.jpg\t1325\t(25, 32)\tf\t988\t1228\t382\t383\t5\t0\t74
113445054@N07\t11763996693_bb46e655f7_o.jpg\t1329\t34\tm\t1384\t1077\t313\t312\t-10\t-15\t112
113445054@N07\t11764047416_d3ea1afc38_o.jpg\t1329\t34\tm\t1942\t988\t580\t580\t5\t0\t53""",
        """user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\ttilt_ang\tfiducial_yaw_angle\tfiducial_score
115321157@N03\t12111738395_a7f715aa4e_o.jpg\t1744\t(4, 6)\tm\t663\t997\t637\t638\t-95\t0\t129
115321157@N03\t12112413505_0aea8e17c6_o.jpg\t1745\t(48, 53)\tm\t505\t846\t433\t433\t-95\t0\t72
115321157@N03\t12112392255_995532c2f0_o.jpg\t1744\t(4, 6)\tm\t517\t1185\t383\t383\t0\t0\t70
115321157@N03\t12112392255_995532c2f0_o.jpg\t1746\t(25, 32)\tm\t2247\t688\t376\t376\t0\t30\t67
115321157@N03\t12112392255_995532c2f0_o.jpg\t1747\t(25, 32)\tm\t1421\t667\t325\t325\t0\t0\t102
115321157@N03\t12111055306_38d54c12ff_o.jpg\t1747\t(25, 32)\tm\t513\t247\t2205\t2201\t-95\t0\t107
115321157@N03\t12120203274_f0390d9f7c_o.jpg\t1748\t(0, 2)\tu\t0\t149\t1813\t2155\t-115\t0\t78
115321157@N03\t12123773476_b75f30a314_o.jpg\t1748\t(0, 2)\tu\t1157\t721\t809\t810\t-100\t45\t20
115321157@N03\t12111034286_4f5bfbacea_o.jpg\t1749\t(25, 32)\tf\t1826\t997\t306\t306\t-90\t0\t89
115321157@N03\t12119809715_efb705d9bf_o.jpg\t1744\t(4, 6)\tm\t640\t596\t1237\t1236\t-100\t30\t46
115321157@N03\t12113086695_1962742774_o.jpg\t1744\t(4, 6)\tm\t704\t809\t1135\t1135\t-100\t0\t101
115321157@N03\t12123096015_ae4d8770fa_o.jpg\t1750\t57\tm\t874\t624\t523\t523\t-20\t-45\t33
115321157@N03\t12123096015_ae4d8770fa_o.jpg\t1748\t(0, 2)\tu\t1091\t1012\t325\t325\t-25\t0\t112
115321157@N03\t12120187433_4df14bb039_o.jpg\t1748\t(0, 2)\tu\t851\t541\t1625\t1626\t-105\t0\t41
115321157@N03\t12110347765_b8bb6fed6e_o.jpg\t1749\t(25, 32)\tf\t213\t0\t744\t622\t-100\t0\t134
115321157@N03\t12110347765_b8bb6fed6e_o.jpg\t1747\t(25, 32)\tm\t246\t346\t684\t614\t-85\t0\t75
115321157@N03\t12120008724_81dc81b103_o.jpg\t1744\t(4, 6)\tm\t441\t879\t1039\t1039\t-105\t30\t97
115321157@N03\t12120183513_070b6c677c_o.jpg\t1747\t(25, 32)\tm\t725\t497\t937\t937\t-70\t30\t128
115321157@N03\t12112793214_c8a93a8aa2_o.jpg\t1744\t(4, 6)\tm\t48\t0\t1232\t960\t-95\t0\t41
115321157@N03\t12122286096_b89c88efc6_o.jpg\t1744\t(4, 6)\tm\t1047\t935\t433\t434\t-105\t30\t104""",
    ]

    user_ids = []
    face_ids = []
    image_names = []

    for fold in folds:
        fold_path = folds_path / f"fold_{folds.index(fold)}_data.txt"
        with open(fold_path, "w") as f:
            f.write(fold)

        for line in fold.split("\n")[1:]:
            user_id, original_image, face_id, age = line.split("\t")[:4]
            user_ids.append(user_id)
            face_ids.append(face_id)
            image_names.append(original_image)

    for user_id, face_id, image_name in zip(user_ids, face_ids, image_names):
        (images_path / user_id).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (816, 816)).save(
            images_path / user_id / f"landmark_aligned_face.{face_id}.{image_name}"
        )

    # Archive and compress the images folder in a tar.gz file
    with tarfile.open(tmp_path / "aligned.tar.gz", "w:gz") as f:
        f.add(images_path, arcname="aligned")

    return images_path, folds_path


def get_adience_instance(tmp_path, train, verbose=False):
    images_path, folds_path = _create_adience_data(tmp_path)

    adience_instance = Adience(
        root=tmp_path,
        train=train,
        ranges=[
            (0, 2),
            (4, 6),
            (8, 13),
            (15, 20),
            (25, 32),
            (38, 43),
            (48, 53),
            (60, 100),
        ],
        test_size=0.2,
        verbose=verbose,
    )

    return adience_instance


@pytest.fixture
def adience_train(tmp_path):
    return get_adience_instance(tmp_path, train=True, verbose=True)


@pytest.fixture
def adience_test(tmp_path):
    return get_adience_instance(tmp_path, train=False, verbose=False)


def test_adience_init(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._check_if_extracted()
        assert adience._check_if_transformed()
        assert adience._check_if_partitioned()
        assert adience._check_input_files()


def test_adience_len(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert len(adience) == len(adience.targets)
        assert len(adience) == len(adience.data)

        adience.targets.append(0)

        with pytest.raises(ValueError):
            len(adience)


def test_adience_getitem(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        for i in range(len(adience)):
            assert isinstance(adience[i][0], Image.Image)
            assert isinstance(adience[i][1], int)
            assert adience[i][1] == adience.targets[i]
            assert np.array(adience[i][0]).ndim == 3

        adience.transform = ToTensor()

        for i in range(len(adience)):
            assert isinstance(adience[i][0], torch.Tensor)
            assert isinstance(adience[i][1], int)
            assert adience[i][1] == adience.targets[i]
            assert len(adience[i][0].shape) == 3

        adience.target_transform = lambda target: np.array(target)
        for i in range(len(adience)):
            assert isinstance(adience[i][0], torch.Tensor)
            assert isinstance(adience[i][1], np.ndarray)
            assert np.array_equal(adience[i][1], adience.targets[i])


def test_assign_range_integers(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("1") == 0
        assert adience._assign_range("5") == 1
        assert adience._assign_range("10") == 2
        assert adience._assign_range("18") == 3
        assert adience._assign_range("30") == 4
        assert adience._assign_range("41") == 5
        assert adience._assign_range("50") == 6
        assert adience._assign_range("70") == 7
        assert adience._assign_range("101") is None


def test_assing_range_tuples(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("(0, 2)") == 0
        assert adience._assign_range("(4, 6)") == 1
        assert adience._assign_range("(8, 13)") == 2
        assert adience._assign_range("(15, 20)") == 3
        assert adience._assign_range("(25, 32)") == 4
        assert adience._assign_range("(38, 43)") == 5
        assert adience._assign_range("(48, 53)") == 6
        assert adience._assign_range("(60, 100)") == 7


def test_assign_range_none(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("None") is None


def test_adience_train_test(adience_train, adience_test):
    assert len(adience_train) != len(adience_test)

    train_labels = [label for _, label in adience_train]
    test_labels = [label for _, label in adience_test]

    assert train_labels != test_labels


def test_image_path_from_row():
    row = {"user_id": "123", "face_id": "456", "original_image": "image.jpg"}
    path = _image_path_from_row(row)
    assert path == "123/landmark_aligned_face.456.image.jpg"


def test_track_progress():
    tar_file_path = "fake.tar.gz"

    try:
        with tarfile.open(tar_file_path, "w:gz") as file:
            for member in _track_progress(file):
                assert isinstance(member, tarfile.TarInfo)

    finally:
        os.remove(tar_file_path)


def test_process_and_split(monkeypatch, tmp_path):
    mock_image_open = Mock(side_effect=lambda _: Image.new("RGB", (128, 128)))
    monkeypatch.setattr("PIL.Image.open", mock_image_open)
    mock_symlink_to = Mock()
    monkeypatch.setattr("pathlib.Path.symlink_to", mock_symlink_to)

    for train in [True, False]:
        adience = get_adience_instance(tmp_path, train=train)

        shutil.rmtree(adience.transformed_images_path_)
        shutil.rmtree(adience.partition_path_)

        initial_open_count = mock_image_open.call_count
        initial_symlink_count = mock_symlink_to.call_count

        adience._process_and_split(adience.folds_)

        assert mock_image_open.call_count == initial_open_count + len(adience.df_)

        if train:
            assert mock_symlink_to.call_count == pytest.approx(
                initial_symlink_count + len(adience.df_) * (1 - adience.test_size),
                abs=1,
            )
        else:
            assert mock_symlink_to.call_count == pytest.approx(
                initial_symlink_count + len(adience.df_) * adience.test_size, abs=1
            )

        adience._process_and_split(adience.folds_)

        assert mock_image_open.call_count == initial_open_count + len(adience.df_)
        if train:
            assert mock_symlink_to.call_count == pytest.approx(
                initial_symlink_count + len(adience.df_) * (1 - adience.test_size),
                abs=1,
            )
        else:
            assert mock_symlink_to.call_count == pytest.approx(
                initial_symlink_count + len(adience.df_) * adience.test_size, abs=1
            )

        shutil.rmtree(adience.transformed_images_path_)

        initial_open_count = mock_image_open.call_count
        initial_symlink_count = mock_symlink_to.call_count

        adience._process_and_split(adience.folds_)

        assert mock_image_open.call_count == initial_open_count + len(adience.df_)

        assert mock_symlink_to.call_count == initial_symlink_count


def test_adience_classes(adience_train, adience_test):
    assert adience_train.classes == adience_test.classes
    assert adience_train.classes == np.unique(adience_train.targets).tolist()
    assert adience_test.classes == np.unique(adience_test.targets).tolist()


def test_check_input_files(adience_train, adience_test, tmp_path):
    assert adience_train._check_input_files()
    assert adience_test._check_input_files()

    adience_train.data_file_path_.unlink()
    assert not adience_train._check_input_files()

    (adience_test.folds_path_ / "fold_0_data.txt").unlink()
    assert not adience_test._check_input_files()

    with pytest.raises(FileNotFoundError):
        Adience(root=Path(tmp_path) / "test", train=True)


def test_check_if_extracted(adience_train, adience_test):
    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    shutil.rmtree(adience_train.images_path_)
    assert not adience_train._check_if_extracted()
    assert not adience_test._check_if_extracted()


def test_check_if_transformed(adience_train, adience_test):
    assert adience_train._check_if_transformed()
    assert adience_test._check_if_transformed()

    shutil.rmtree(adience_train.transformed_images_path_)
    assert not adience_train._check_if_transformed()
    assert not adience_test._check_if_transformed()


def test_check_if_partitioned(adience_train, adience_test):
    assert adience_train._check_if_partitioned()
    assert adience_test._check_if_partitioned()

    shutil.rmtree(adience_train.partition_path_)
    assert not adience_train._check_if_partitioned()
    assert not adience_test._check_if_partitioned()


def test_extract_data(adience_train, adience_test):
    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    adience_train._extract_data()
    adience_test._extract_data()

    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    shutil.rmtree(adience_train.images_path_)

    assert not adience_train._check_if_extracted()
    assert not adience_test._check_if_extracted()

    adience_train._extract_data()

    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()
