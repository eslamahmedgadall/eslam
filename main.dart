import 'package:flutter/material.dart';

import 'package:splash_project/splash/presentation/views/splash_view.dart';

void main() {
  runApp(const HomeSplashApp());
}

class HomeSplashApp extends StatelessWidget {
  const HomeSplashApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: SplashView(),
    );
  }
}
