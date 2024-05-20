import 'package:flutter/material.dart';
import 'package:splash_project/table/presentetion/hexa_color.dart';

Widget myDivider(context) {
  return Container(
    height: 1,
    color: Colors.grey.shade400,
    width: MediaQuery.of(context).size.width,
  );
}

Widget addbutton(
    {required IconData icon,
    required Color color,
    required VoidCallback onpressed}) {
  return Container(
    width: 20,
    child: MaterialButton(
      padding: EdgeInsets.zero,
      height: 20,
      onPressed: onpressed,
      color: color,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(6)),
      child: Icon(
        icon,
        color: Colors.white,
        size: 20,
      ),
    ),
  );
}

Widget userbutton(
  context, {
  required String text,
  // String? imageroute,
  required VoidCallback onpressed,
}) {
  return Container(
    width: MediaQuery.of(context).size.width / 2.5,
    child: MaterialButton(
      elevation: 6,
      padding: EdgeInsets.zero,
      height: 45,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(13)),
      color: HexColor('#45A2CF'),
      textColor: Colors.white,
      onPressed: onpressed,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Image(
          //   image: AssetImage(imageroute ?? ''),
          //   height: 15,
          //   width: 15,
          //   // color: Colors.white,
          // ),
          SizedBox(
            width: 5,
          ),
          Text(
            text,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    ),
  );
}

Widget myText(context,
    {required String data,
    required Color? color,
    required double? fontSize,
    required FontWeight? fontWeight}) {
  return Text(
    data,
    style: TextStyle(fontSize: fontSize, fontWeight: fontWeight, color: color),
  );
}

Widget textTitle(
  context, {
  required String data,
  Color? color,
}) {
  return Text(
    data,
    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: color),
  );
}
