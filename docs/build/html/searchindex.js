Search.setIndex({docnames:["concepts","getting-started","index","installation","sdk/index","sdk/modalic","sdk/modalic.client","sdk/modalic.client.proto","sdk/modalic.client.utils","sdk/modalic.config","sdk/modalic.data","sdk/modalic.logging","sdk/modalic.server","sdk/modalic.simulation","sdk/modalic.storage","sdk/modalic.utils","sdk/modules","tutorials-and-examples/index","tutorials-and-examples/quickstart-pytorch"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["concepts.rst","getting-started.rst","index.rst","installation.rst","sdk/index.rst","sdk/modalic.rst","sdk/modalic.client.rst","sdk/modalic.client.proto.rst","sdk/modalic.client.utils.rst","sdk/modalic.config.rst","sdk/modalic.data.rst","sdk/modalic.logging.rst","sdk/modalic.server.rst","sdk/modalic.simulation.rst","sdk/modalic.storage.rst","sdk/modalic.utils.rst","sdk/modules.rst","tutorials-and-examples/index.rst","tutorials-and-examples/quickstart-pytorch.rst"],objects:{"":[[5,0,0,"-","modalic"]],"modalic.PytorchClient":[[4,2,1,"","dtype"],[4,2,1,"","loss"],[4,2,1,"","model_shape"],[4,2,1,"","round_id"],[4,3,1,"","train"]],"modalic.client":[[6,1,1,"","Trainer"],[6,0,0,"-","grpc_client"],[7,0,0,"-","proto"],[6,0,0,"-","pytorch_client"],[6,0,0,"-","tensorflow_client"],[6,0,0,"-","trainer"],[8,0,0,"-","utils"]],"modalic.client.Trainer":[[6,4,1,"","model"],[6,3,1,"","train"]],"modalic.client.grpc_client":[[6,1,1,"","CommunicationLayer"],[6,1,1,"","Communicator"]],"modalic.client.grpc_client.CommunicationLayer":[[6,3,1,"","get_global_model"],[6,3,1,"","update"]],"modalic.client.grpc_client.Communicator":[[6,3,1,"","get_global_model"],[6,3,1,"","grpc_connection"],[6,3,1,"","update"]],"modalic.client.proto":[[7,0,0,"-","mosaic_pb2"],[7,0,0,"-","mosaic_pb2_grpc"]],"modalic.client.proto.mosaic_pb2_grpc":[[7,1,1,"","Communication"],[7,1,1,"","CommunicationServicer"],[7,1,1,"","CommunicationStub"],[7,5,1,"","add_CommunicationServicer_to_server"]],"modalic.client.proto.mosaic_pb2_grpc.Communication":[[7,3,1,"","GetGlobalModel"],[7,3,1,"","Update"]],"modalic.client.proto.mosaic_pb2_grpc.CommunicationServicer":[[7,3,1,"","GetGlobalModel"],[7,3,1,"","Update"]],"modalic.client.pytorch_client":[[6,1,1,"","PytorchClient"]],"modalic.client.pytorch_client.PytorchClient":[[6,2,1,"","dtype"],[6,2,1,"","loss"],[6,2,1,"","model_shape"],[6,2,1,"","round_id"],[6,3,1,"","train"]],"modalic.client.trainer":[[6,1,1,"","Trainer"]],"modalic.client.trainer.Trainer":[[6,4,1,"","model"],[6,3,1,"","train"]],"modalic.client.utils":[[8,0,0,"-","communication"],[8,0,0,"-","decor"],[8,0,0,"-","torch_utils"]],"modalic.config":[[9,1,1,"","Conf"],[9,0,0,"-","config"]],"modalic.config.Conf":[[9,4,1,"","certificates"],[9,4,1,"","client_id"],[9,3,1,"","create_conf"],[9,4,1,"","data_type"],[9,3,1,"","from_toml"],[9,4,1,"","participants"],[9,4,1,"","server_address"],[9,3,1,"","set_params"],[9,4,1,"","timeout"],[9,4,1,"","training_rounds"]],"modalic.config.config":[[9,1,1,"","Conf"]],"modalic.config.config.Conf":[[9,4,1,"","certificates"],[9,4,1,"","client_id"],[9,3,1,"","create_conf"],[9,4,1,"","data_type"],[9,3,1,"","from_toml"],[9,4,1,"","participants"],[9,4,1,"","server_address"],[9,3,1,"","set_params"],[9,4,1,"","timeout"],[9,4,1,"","training_rounds"]],"modalic.data":[[10,0,0,"-","data"],[10,0,0,"-","misc"]],"modalic.data.data":[[10,1,1,"","DataPool"]],"modalic.data.misc":[[10,5,1,"","get_dataset_length"]],"modalic.logging":[[11,1,1,"","Monitor"],[11,0,0,"-","logging"],[11,0,0,"-","monitor"]],"modalic.logging.logging":[[11,1,1,"","CustomFormatter"]],"modalic.logging.logging.CustomFormatter":[[11,4,1,"","bold_red"],[11,3,1,"","custom_format"],[11,3,1,"","custom_format_str"],[11,4,1,"","faint"],[11,3,1,"","format"],[11,4,1,"","green"],[11,4,1,"","red"],[11,4,1,"","reset"],[11,4,1,"","yellow"]],"modalic.logging.monitor":[[11,1,1,"","Monitor"]],"modalic.server":[[12,0,0,"-","api"],[12,0,0,"-","server"]],"modalic.server.api":[[12,6,1,"","ServerBinaryNotFound"],[12,5,1,"","find_bin_path"]],"modalic.server.server":[[12,5,1,"","run_server"]],"modalic.simulation":[[13,1,1,"","ClientPool"],[13,0,0,"-","pool"]],"modalic.simulation.ClientPool":[[13,3,1,"","exec_single_thread"],[13,3,1,"","run"],[13,3,1,"","spawn_pool"]],"modalic.simulation.pool":[[13,1,1,"","ClientPool"]],"modalic.simulation.pool.ClientPool":[[13,3,1,"","exec_single_thread"],[13,3,1,"","run"],[13,3,1,"","spawn_pool"]],"modalic.storage":[[14,1,1,"","Storage"],[14,0,0,"-","store"]],"modalic.storage.Storage":[[14,3,1,"","upload"]],"modalic.storage.store":[[14,1,1,"","Storage"]],"modalic.storage.store.Storage":[[14,3,1,"","upload"]],"modalic.utils":[[15,0,0,"-","convert"],[15,0,0,"-","protocol"],[15,0,0,"-","serde"],[15,0,0,"-","shared"]],"modalic.utils.protocol":[[15,5,1,"","parameters_from_proto"],[15,5,1,"","parameters_to_proto"],[15,5,1,"","process_meta_to_proto"],[15,5,1,"","to_meta"]],"modalic.utils.serde":[[15,5,1,"","get_shape"],[15,5,1,"","parameters_to_weights"],[15,5,1,"","weights_to_parameters"]],"modalic.utils.shared":[[15,1,1,"","Parameters"],[15,1,1,"","ProcessMeta"]],"modalic.utils.shared.Parameters":[[15,4,1,"","data_type"],[15,4,1,"","model_version"],[15,4,1,"","tensor"]],"modalic.utils.shared.ProcessMeta":[[15,4,1,"","loss"],[15,4,1,"","round_id"]],modalic:[[4,1,1,"","PytorchClient"],[4,0,0,"-","client"],[9,0,0,"-","config"],[10,0,0,"-","data"],[11,0,0,"-","logging"],[4,5,1,"","run_server"],[12,0,0,"-","server"],[13,0,0,"-","simulation"],[14,0,0,"-","storage"],[4,5,1,"","train"],[15,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","property","Python property"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"],"6":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:property","3":"py:method","4":"py:attribute","5":"py:function","6":"py:exception"},terms:{"0":[4,6,8,9,10,13],"0m":11,"1":[4,6,10,13,14],"1m":11,"20m":11,"2m":11,"3":3,"31":11,"32m":11,"33":11,"5":6,"536870912":6,"8":3,"8080":[4,8,9],"9":[],"abstract":[4,6],"byte":[6,15],"class":[4,6,7,9,10,11,13,14,15],"default":[6,9,11],"float":[6,9,15],"function":[1,4,6,8,13],"import":[1,6],"int":[4,6,9,10,11,13,14,15],"return":[6,7,10,11,12,15],"static":[6,7],"true":11,"while":[4,8,11],As:3,But:3,For:1,If:10,It:1,The:[1,9],There:1,abc:6,abil:1,about:[2,15],abov:3,access:14,achiev:0,action:15,add_communicationservicer_to_serv:7,address:6,after:9,aggreg:[0,6,9,12],aim:[1,2],alia:[],all:[4,6],alwai:9,an:[1,2,6],anaconda3:[],anaconda:[],ananconda:1,ani:[4,6,9,10,13,14,15],api:[1,4,5,6,16],applic:12,ar:[0,1,3,13],architectur:6,arg:[6,14],associ:7,attributeerror:[4,6],avail:3,base:[6,7,9,10,11,12,13,14,15],basic:1,below:17,between:6,binari:[12,15],bold_r:11,bool:6,bucket:14,buffer:7,bunch:13,call:10,call_credenti:7,can:[4,6,17],cannot:10,central:0,certif:[4,8,9],cfg_path:[4,12],channel:[6,7,9],channel_credenti:7,classmethod:9,client:[0,5,9,10,11,13,16],client_id:[4,6,8,9,10],clientpool:13,code:[3,7],collabor:0,color:11,come:[1,3],comment:7,commun:[4,5,6,7],communicatio:6,communicationlay:6,communicationservic:7,communicationstub:[6,7],compat:[4,6],complet:6,compress:7,concept:2,concern:[4,6],concurr:13,conda:1,conf:[4,6,8,9,14],config:[4,5,8,16],configur:[4,6,9,14],conig:9,connect:6,construct:9,contain:[4,6,9,14],content:16,context:7,control:1,convert:[5,16],coordin:0,correspond:7,creat:1,create_conf:9,custom:[10,11],custom_format:11,custom_format_str:11,customformatt:11,daniel:[],data:[0,4,5,6,7,9,14,15,16],data_typ:[4,8,9,15],datapool:10,dataset:[4,6,10],datefmt:11,de:[6,9],decor:[5,6],defin:[1,7,9],deseri:[6,15],determin:6,develop:1,devic:9,dict:[4,6,9,11,14],dictionari:14,differ:1,distribut:[4,6,10],doc:[],document:7,doe:1,done:1,dtype:[4,6,15],dure:6,each:[0,9],edg:9,en:[],enabl:[1,4,6,14],encrypt:6,endpoint:[1,9,13],ensur:6,entir:1,entiti:0,env:[],environ:1,error:12,establish:[6,9],everyth:1,exampl:[2,4,6,8,9],except:12,exchang:0,exec_single_thread:13,execut:13,extern:9,f32:[4,8,9],faint:11,fals:[6,7],featur:6,feder:[1,2,4,6,8,9,12,13,17],file:[7,9,12,15],find:[12,17],find_bin_path:12,fl:1,flop:[],fmt:11,focus:0,format:[11,15],formatt:11,found:12,from:[6,9],from_toml:9,full:1,fulli:3,further:1,futur:13,gener:[1,7],get:[2,6,11],get_dataset_length:10,get_global_model:6,get_shap:15,getglobalmodel:7,given:9,global:[6,7],globalmeta:7,green:11,grpc:[6,9,15],grpc_client:[5,16],grpc_connect:6,guid:1,ha:[1,4,6],handl:[7,10],handler:10,hold:[6,13],home:[],html:[],http:[],id:[4,6],identifi:[4,6,9],immedi:0,implement:6,impli:1,independet:10,index:[],individu:13,inform:1,inherit:6,input:[4,6],insecur:7,instal:2,instanti:14,instead:0,integ:6,integr:1,intend:0,io:[],ip:6,its:15,keep:[],kei:2,kind:1,known:15,kwarg:6,latest:6,launch:13,layer:[4,6,14],learn:[1,2,4,6,8,9,12,13,17],len:10,length:[9,10,14],let:1,lib:[],list:[6,13,15],local:[0,6,7],log:[5,6,16],log_nam:11,logback:6,logrecord:11,loop:[4,6],loss:[4,6,15],machin:[0,1],main:[1,13],mainli:9,major:1,manag:13,max_message_length:6,maximum:6,messag:6,meta:[7,15],metadata:[7,14],misc:[5,16],miss:7,mlop:[2,17],modal:[1,2,17],model:[1,4,6,7,9,15],model_shap:[4,6],model_vers:15,modul:16,monitor:[5,16],mosaic_pb2:[5,6,15],mosaic_pb2_grpc:[5,6],msg:15,multipl:0,multipli:[4,8],must:9,name:[11,13],ndarrai:[6,15],neg:9,nn:6,non:9,none:[4,6,7,9,11,12,13,14],note:1,notimpl:6,now:1,np:15,num_client:13,num_work:13,number:[6,9,13,17],numpi:[6,15],object:[0,1,4,6,7,9,10,11,13,14,15],object_nam:14,offer:1,one:1,onli:1,open:[1,2],option:[4,6,7,9,10,13,14],order:6,org:[],orient:1,otherwis:1,overwrit:9,packag:16,paradigm:1,parallel:13,param:[6,9],paramet:[4,6,9,10,11,13,15],parameters_from_proto:15,parameters_to_proto:15,parameters_to_weight:15,particip:[1,4,6,8,9],path:[9,12],perform:[2,4,8,9],period:9,pip:[1,3],platform:[2,17],pleas:1,pool:[5,16],possibl:[1,3],print:11,problem:[0,1],procedur:[1,4,8],process:[1,4,6,7,9,15],process_meta_to_proto:15,processmeta:15,produc:9,program:1,project:[],properti:[4,6],proto:[5,6],protobuf:[7,15],protocol:[5,6,7,16],provid:[0,2,6],py:[],pypi:3,python3:[],python:2,pytorch:[6,10,17],pytorch_cli:[5,16],pytorchcli:[6,13],quick:1,quickstart:17,rais:[4,6],rather:1,raw:0,read:[2,15],recommend:1,record:11,red:11,refer:2,regard:[1,7,9],relat:12,releas:[],remot:7,repres:1,request:[6,7],requir:[3,9],reset:11,retri:6,root_certif:6,round:[6,9],round_id:[4,6,15],rpc:7,run:[1,4,6,12,13],run_:[],run_serv:[4,12],s3:14,s:[0,1],safe:6,sampl:6,script:15,sdk:[2,3],second:9,secur:9,send:[6,7],separ:13,serd:[5,16],serial:[6,9,15],serv:[1,4,6],server:[0,3,5,6,7,9,16],server_address:[4,6,8,9],serverbinarynotfound:12,servermessag:7,servic:[0,7],set:[0,4,6],set_param:9,setglobalmeta:7,setup:1,shape:[6,15],share:[5,16],should:9,side:1,simpl:[1,4,6],simpli:1,simul:[5,9,10,16],singl:[4,6,9,13],site:[],size:6,solv:0,some:[6,9],soon:1,sourc:[1,2,4,6,7,8,9,10,11,12,13,14,15],space:1,spawn_pool:13,stabl:[],stack:1,stake:6,start:2,state:9,storag:[5,16],store:[0,4,5,6,9,16],str:[4,6,9,11,12,13,14,15],string:[9,11],stub:6,style:11,submodul:[5,16],subpackag:16,target:7,task:1,tensor:15,tensorflow:1,tensorflow_cli:[5,16],tensorflowcli:13,thi:3,thread:[6,13],threadpoolexecutor:13,threat:10,thrown:12,time:[4,8],timeout:[4,7,8,9],tl:9,to_meta:15,toml:9,toolkit:1,torch:6,torch_util:[5,6],train:[6,8,9,11],trainer:[4,5,16],training_round:[4,8,9],transfer:0,tupl:6,tutori:2,two:1,type:[1,6,9],under:0,underli:[4,8],uniqu:[4,6,9],up:3,updat:[0,6,7],upload:14,us:[0,2,9,13,17],user:[],util:[5,6,16],valid:11,valu:9,version:6,via:6,virtual:1,visit:1,wa:6,wait:9,wait_for_readi:7,want:13,weight:15,weights_to_paramet:15,what:1,when:12,where:0,which:[4,6,9,10,12,13],whole:[4,6,13],within:[1,4,6],worker:13,www:[],x1b:11,yellow:11,yet:3,you:[13,17]},titles:["Concepts","Getting Started","Documentation","Installing Modalic","Python SDK Reference","modalic package","modalic.client package","modalic.client.proto package","modalic.client.utils package","modalic.config package","modalic.data package","modalic.logging package","modalic.server package","modalic.simulation package","modalic.storage package","modalic.utils package","modalic","Tutorials and Examples","Quickstart PyTorch"],titleterms:{aggreg:4,api:12,client:[1,4,6,7,8],commun:8,concept:0,config:9,content:[2,5,6,7,8,9,10,11,12,13,14,15,17],convert:15,data:10,decor:[4,8],document:2,entrypoint:1,exampl:17,feder:0,framework:1,from:3,get:1,grpc_client:6,instal:[1,3],learn:0,log:11,misc:10,modal:[3,4,5,6,7,8,9,10,11,12,13,14,15,16],modul:[4,5,6,7,8,9,10,11,12,13,14,15],monitor:11,mosaic_pb2:7,mosaic_pb2_grpc:7,packag:[5,6,7,8,9,10,11,12,13,14,15],pool:13,proto:7,protocol:15,python:[1,3,4],pytorch:[1,4,18],pytorch_cli:6,pytorchcli:4,quickstart:18,refer:4,releas:3,sdk:[1,4],serd:15,server:[4,12],share:15,simul:13,sourc:3,stabl:3,start:1,storag:14,store:14,submodul:[6,7,8,9,10,11,12,13,14,15],subpackag:[5,6],support:1,tensorflow_cli:6,torch_util:8,train:4,trainer:6,tutori:17,util:[8,15],version:3}})