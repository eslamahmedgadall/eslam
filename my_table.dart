import 'package:flutter/material.dart';

import 'package:splash_project/table/presentetion/hexa_color.dart';
import 'package:splash_project/utils/assets.dart';
import 'package:splash_project/table/presentetion/view/widgets/components_widgets.dart';

class MyTaple extends StatelessWidget {
  const MyTaple({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showModalBottomSheet(
            isScrollControlled: true,
            shape: const RoundedRectangleBorder(
                borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(20),
                    topRight: Radius.circular(20))),
            context: context,
            builder: (context) {
              return Container(
                width: double.infinity,
                child: Padding(
                  padding: EdgeInsets.only(
                      bottom: MediaQuery.of(context).viewInsets.bottom),
                  child: Padding(
                    padding: const EdgeInsets.all(15),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        myText(context,
                            data: 'Add New medcine: ',
                            color: HexColor('#29256E'),
                            fontSize: 16,
                            fontWeight: FontWeight.w600),
                        const SizedBox(
                          height: 5,
                        ),
                        myDivider(context),
                        SizedBox(
                          height: MediaQuery.of(context).size.height / 50,
                        ),
                        Column(
                          children: [
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 30),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  myText(context,
                                      data: 'Name : ',
                                      color: HexColor('#022247'),
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700),
                                  const SizedBox(
                                    width: 3,
                                  ),
                                  Container(
                                    height: 25,
                                    width:
                                        MediaQuery.of(context).size.width / 1.7,
                                    child: TextFormField(
                                      keyboardType: TextInputType.text,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(
                              height: 10,
                            ),
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 30),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  myText(context,
                                      data: 'price : ',
                                      color: HexColor('#022247'),
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700),
                                  const SizedBox(
                                    width: 3,
                                  ),
                                  Container(
                                    height: 25,
                                    width:
                                        MediaQuery.of(context).size.width / 1.7,
                                    child: TextFormField(
                                      keyboardType: TextInputType.number,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(
                              height: 10,
                            ),
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(horizontal: 30),
                              child: Row(
                                children: [
                                  myText(context,
                                      data: 'Quantity : ',
                                      color: HexColor('#022247'),
                                      fontSize: 15,
                                      fontWeight: FontWeight.w700),
                                  addbutton(
                                      icon: Icons.add,
                                      color: HexColor('#80B2CA'),
                                      onpressed: () {}),
                                  const SizedBox(
                                    width: 7,
                                  ),
                                  Text(
                                    '9',
                                    style: TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.w600,
                                      color: HexColor('#6A6A6A'),
                                    ),
                                  ),
                                  const SizedBox(
                                    width: 7,
                                  ),
                                  addbutton(
                                      icon: Icons.remove,
                                      color: HexColor('#455A64'),
                                      onpressed: () {}),
                                  const SizedBox(
                                    height: 30,
                                  ),
                                ],
                              ),
                            ),
                            Center(
                              child: userbutton(context, text: 'Save Changes',
                                  onpressed: () {
                                Navigator.pop(context);
                              }),
                            ),
                          ],
                        )
                      ],
                    ),
                  ),
                ),
              );
            },
          );
        },
        child: Icon(Icons.add),
      ),
      body: Column(
        children: [
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 20, vertical: 15),
            child: Row(
              children: [],
            ),
          ),
          const SizedBox(
            height: 10,
          ),
          const SizedBox(
            height: 17,
          ),
          DataTable(
              border: TableBorder.all(
                  color: Colors.grey.withOpacity(.1),
                  style: BorderStyle.solid,
                  width: 1,
                  borderRadius: BorderRadius.circular(20)),
              decoration: BoxDecoration(
                  color: Colors.white, borderRadius: BorderRadius.circular(20)),
              columnSpacing: 25,
              columns: const [
                DataColumn(
                    label: Text(
                  'Img',
                  style: TextStyle(fontSize: 12),
                )),
                DataColumn(
                    label: Text(
                  'Name',
                  style: TextStyle(fontSize: 12),
                )),
                DataColumn(
                    label: Text(
                  'Price',
                  style: TextStyle(fontSize: 12),
                )),
                DataColumn(
                    label: Text(
                  'Quantity',
                  style: TextStyle(fontSize: 12),
                )),
                DataColumn(
                    label: Text(
                  'Edit',
                  style: TextStyle(fontSize: 12),
                )),
              ],
              rows: [
                DataRow(cells: [
                  const DataCell(Image(
                    image: AssetImage('assets/photos/5000347029479 3.png'),
                    width: 40,
                    height: 30,
                  )),
                  const DataCell(Text(
                    'Panadol',
                    style: TextStyle(fontSize: 10),
                  )),
                  const DataCell(Text(
                    '30',
                    style: TextStyle(fontSize: 10),
                  )),
                  const DataCell(Text(
                    '6',
                    style: TextStyle(fontSize: 10),
                  )),
                  DataCell(Row(
                    children: [
                      InkWell(
                        onTap: () {
                          showModalBottomSheet(
                            isScrollControlled: true,
                            shape: const RoundedRectangleBorder(
                                borderRadius: BorderRadius.only(
                                    topLeft: Radius.circular(20),
                                    topRight: Radius.circular(20))),
                            context: context,
                            builder: (context) {
                              return Container(
                                width: double.infinity,
                                child: Padding(
                                  padding: EdgeInsets.only(
                                      bottom: MediaQuery.of(context)
                                          .viewInsets
                                          .bottom),
                                  child: Padding(
                                    padding: const EdgeInsets.all(15),
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Row(
                                          children: [
                                            myText(context,
                                                data: 'Update a medcine: ',
                                                color: HexColor('#29256E'),
                                                fontSize: 16,
                                                fontWeight: FontWeight.w600),
                                            const SizedBox(
                                              width: 24,
                                            ),
                                            myText(context,
                                                data: 'amresol',
                                                color: HexColor('#6A6A6A'),
                                                fontSize: 16,
                                                fontWeight: FontWeight.w600),
                                          ],
                                        ),
                                        const SizedBox(
                                          height: 5,
                                        ),
                                        myDivider(context),
                                        SizedBox(
                                          height: MediaQuery.of(context)
                                                  .size
                                                  .height /
                                              50,
                                        ),
                                        Column(
                                          children: [
                                            Padding(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                      horizontal: 30),
                                              child: Row(
                                                crossAxisAlignment:
                                                    CrossAxisAlignment.center,
                                                children: [
                                                  myText(context,
                                                      data: 'price : ',
                                                      color:
                                                          HexColor('#022247'),
                                                      fontSize: 15,
                                                      fontWeight:
                                                          FontWeight.w700),
                                                  const SizedBox(
                                                    width: 3,
                                                  ),
                                                  Container(
                                                    height: 25,
                                                    width:
                                                        MediaQuery.of(context)
                                                                .size
                                                                .width /
                                                            1.7,
                                                    child: TextFormField(
                                                      keyboardType:
                                                          TextInputType.number,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                            const SizedBox(
                                              height: 10,
                                            ),
                                            Padding(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                      horizontal: 30),
                                              child: Row(
                                                children: [
                                                  myText(context,
                                                      data: 'Quantity : ',
                                                      color:
                                                          HexColor('#022247'),
                                                      fontSize: 15,
                                                      fontWeight:
                                                          FontWeight.w700),
                                                  addbutton(
                                                      icon: Icons.add,
                                                      color:
                                                          HexColor('#80B2CA'),
                                                      onpressed: () {}),
                                                  const SizedBox(
                                                    width: 7,
                                                  ),
                                                  Text(
                                                    '9',
                                                    style: TextStyle(
                                                      fontSize: 15,
                                                      fontWeight:
                                                          FontWeight.w600,
                                                      color:
                                                          HexColor('#6A6A6A'),
                                                    ),
                                                  ),
                                                  const SizedBox(
                                                    width: 7,
                                                  ),
                                                  addbutton(
                                                      icon: Icons.remove,
                                                      color:
                                                          HexColor('#455A64'),
                                                      onpressed: () {}),
                                                  const SizedBox(
                                                    height: 30,
                                                  ),
                                                ],
                                              ),
                                            ),
                                            Center(
                                              child: userbutton(context,
                                                  text: 'Save Changes',
                                                  onpressed: () {
                                                Navigator.pop(context);
                                              }),
                                            ),
                                          ],
                                        )
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          );
                        },
                        child: const Image(
                          image: AssetImage(
                            'assets/photos/editing.png',
                          ),
                          color: Colors.blue,
                          height: 25,
                        ),
                      ),
                      const SizedBox(
                        width: 7,
                      ),
                      InkWell(
                        onTap: () {},
                        child: const Image(
                          image: AssetImage(
                            'assets/photos/delete.png',
                          ),
                          color: Colors.red,
                          height: 25,
                        ),
                      ),
                    ],
                  )),
                ]),
              ]),
          const SizedBox(
            height: 20,
          ),
          const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                width: 4,
              ),
            ],
          )
        ],
      ),
    );
  }
}
