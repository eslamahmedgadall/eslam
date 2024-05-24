import 'package:flutter/material.dart';
import 'package:splash_project/splash/presentation/views/widgets/splash_view_body.dart';

class SplashView extends StatelessWidget {
  const SplashView({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
          gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              //transform: GradientRotation(m),
              colors: [
            Color.fromARGB(255, 159, 207, 245),
            Colors.white,
            Color.fromARGB(255, 159, 207, 245),
          ])),
      child: const Scaffold(
        backgroundColor: Colors.transparent,
        body: SplashViewBody(),
      ),
    );
  }
}
