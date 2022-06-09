a=1
for i in gestnet_la_tb_tc_*_200.pt; do
  new=$(printf "gestnet_la_tb_tc_%d_200.pt" "$a") #04 pad to length of 4
  mv $i $new
  let a=a+1
done
